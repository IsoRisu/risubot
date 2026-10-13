"""
localanalyzer.py - offline summary step of profiler.py.

No LLM and no network at analysis time. What it does:

    1. keep the user's prose messages (links, code, emoji tokens stripped)
    2. embed them with a multilingual sentence-transformers model (CPU, PyTorch)
       -> understands Finnish and English, groups messages by meaning
    3. cluster the embeddings (small spherical k-means, numpy only)
    4. label each cluster with its most distinctive words (c-TF-IDF, as in BERTopic)
    5. write the summary from templates using those topics + the style features

Requires `sentence-transformers` (which pulls in PyTorch). If it isn't installed
`_get_model` raises with a clear message; there is no template fallback.

Called from profiler.py as  analyze(records, features, cfg) -> dict  (blocking;
profiler runs it in a worker thread).
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Any, Optional

import numpy as np

import profiler as _p

_model = None
_model_key: Optional[tuple] = None        # (name, device), so a device change reloads
_lock = threading.Lock()


def _get_model(name: str, device: str):
    """Load the embedding model once per (name, device). Raises if it can't be loaded."""
    global _model, _model_key
    key = (name, device)
    with _lock:
        if _model is not None and _model_key == key:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "local profiling needs sentence-transformers: run `poetry install -E local-profiler`"
            ) from e
        _model = SentenceTransformer(name, device=device)
        _model_key = key
        return _model


# ---------- text selection ----------------------------------------------------

def _prose(text: str) -> str:
    return " ".join(_p._NOISE_RE.sub(" ", text).split())


def _select_texts(records, cap: Optional[int]) -> list[str]:
    """Prose messages (>= 3 words once links/code/emoji are removed), oldest first.

    If there are more than `cap`, take an even spread across the whole time range
    rather than only the newest ones, so a full-history scan describes the whole
    history. cap=None keeps everything (embedding tens of thousands of messages on a
    CPU takes minutes)."""
    texts = []
    for r in sorted(records, key=lambda r: r.timestamp):
        t = _prose(r.content)
        if len(_p._WORD_RE.findall(t)) >= 3:
            texts.append(t)
    if cap is not None and len(texts) > cap:
        idx = np.linspace(0, len(texts) - 1, cap).round().astype(int)
        texts = [texts[i] for i in idx]
    if len(texts) > 5000:
        print(f"[profiler] embedding {len(texts)} messages (batched); this can take a few minutes.")
    return texts


# ---------- clustering --------------------------------------------------------

def _kmeans_once(x: np.ndarray, k: int, iters: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """One run of spherical k-means (cosine) on L2-normalised rows, k-means++ init.
    Returns (labels, centers, total similarity to assigned centre: higher is better)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = [x[rng.integers(n)]]
    for _ in range(1, k):
        dist = np.clip(1.0 - np.max(x @ np.array(centers).T, axis=1), 0, None) ** 2
        total = dist.sum()
        centers.append(x[rng.choice(n, p=dist / total)] if total > 0 else x[rng.integers(n)])
    c = np.array(centers)
    labels = np.full(n, -1)
    for _ in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = x[labels == j]
            if len(members):
                v = members.mean(axis=0)
                c[j] = v / (np.linalg.norm(v) or 1.0)
    return labels, c, float(np.max(x @ c.T, axis=1).sum())


def _kmeans(x: np.ndarray, k: int, iters: int = 25,
            restarts: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Best of several k-means runs. A single start can get stuck and merge two topics.

    restarts=None picks a budget based on the corpus size: per-run cost grows with n,
    so small corpora can afford many restarts and large ones cannot. Also early-stops
    when a run already fits the data near-perfectly (score close to n)."""
    n = len(x)
    if restarts is None:
        restarts = 10 if n < 150 else 5 if n < 1000 else 3
    best: Optional[tuple[np.ndarray, np.ndarray]] = None
    best_score = -1.0
    good_enough = 0.92 * n
    for seed in range(restarts):
        labels, centers, score = _kmeans_once(x, k, iters, seed)
        if score > best_score:
            best, best_score = (labels, centers), score
        if score >= good_enough:
            break
    return best  # type: ignore[return-value]


def _label_clusters(texts: list[str], labels: np.ndarray, k: int, top_n: int = 3) -> list[list[str]]:
    """Most distinctive words per cluster: c-TF-IDF = tf(word, cluster) * log(1 + A / freq(word))."""
    counts = [Counter() for _ in range(k)]
    for text, lab in zip(texts, labels):
        for w in _p._WORD_RE.findall(text.lower()):
            if len(w) > 2 and w not in _p._STOPWORDS:
                counts[lab][w] += 1
    total = Counter()
    for c in counts:
        total.update(c)
    avg = max(1.0, sum(total.values()) / k)
    out = []
    for c in counts:
        size = max(1, sum(c.values()))
        scored = {w: (n / size) * math.log(1 + avg / total[w]) for w, n in c.items() if n >= 2}
        if not scored:   # tiny cluster: accept words seen once
            scored = {w: (n / size) * math.log(1 + avg / total[w]) for w, n in c.items()}
        out.append([w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:top_n]])
    return out


def _cluster_topics(texts: list[str], model, cfg) -> tuple[list[str], list[str]]:
    """Returns (topic tags, representative messages), biggest clusters first."""
    x = np.asarray(model.encode(texts, batch_size=64, normalize_embeddings=True,
                                show_progress_bar=False), dtype=np.float32)
    k = max(2, min(cfg.local_max_clusters, len(texts) // 25))
    labels, centers = _kmeans(x, k)
    words = _label_clusters(texts, labels, k)

    topics, reps = [], []
    for j in sorted(range(k), key=lambda j: -int((labels == j).sum())):
        idx = np.flatnonzero(labels == j)
        if len(idx) < max(5, 0.05 * len(texts)) or not words[j]:
            continue                                # too small to be a recurring topic
        tag = " ".join(words[j][:2])
        if tag not in topics:
            topics.append(tag)
            reps.append(texts[idx[int(np.argmax(x[idx] @ centers[j]))]])   # message closest to the centre
    return topics[:5], reps


# ---------- summary -----------------------------------------------------------

def _describe(features: dict[str, Any], topics: list[str]) -> str:
    n = max(1, features.get("message_count", 1))
    avg = features.get("avg_length", 0)
    size = "short" if avg < 40 else "medium-length" if avg < 120 else "long"
    traits = []
    if features.get("emoji_rate", 0) > 0.3:
        traits.append("uses a lot of emoji")
    if features.get("question_per_msg", 0) > 0.3:
        traits.append("asks many questions")
    if features.get("urls", 0) / n > 0.1:
        traits.append("shares links often")
    if features.get("code_blocks", 0) > 0:
        traits.append("posts code")
    s = f"Writes {size} messages in a {features.get('tone', 'neutral')} tone"
    s += (", " + ", ".join(traits) if traits else "") + "."
    if topics:
        s += " Often talks about: " + "; ".join(topics[:3]) + "."
    return s


# ---------- entry point -------------------------------------------------------

def analyze(records, features: dict[str, Any], cfg) -> dict[str, Any]:
    topics: list[str] = []
    samples: list[str] = []

    texts = _select_texts(records, cfg.local_max_messages)
    if len(texts) >= cfg.local_min_for_clusters:
        model = _get_model(cfg.local_model, cfg.local_device)
        try:
            topics, samples = _cluster_topics(texts, model, cfg)
        except Exception as e:
            print(f"[profiler] clustering failed, using frequent terms: {type(e).__name__}: {e}")
            topics, samples = [], []

    terms = features.get("top_terms", [])
    if not topics:
        topics = terms[:5]

    # interests: the head word of each topic first, then overall frequent terms
    interests: list[str] = []
    for w in [t.split()[0] for t in topics] + terms:
        if w and w not in interests:
            interests.append(w)
    return {
        "summary": _describe(features, topics),
        "interests": interests[:7],
        "topics": topics,
        "samples": samples,
    }