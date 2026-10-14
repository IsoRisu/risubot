# llmclient.py
# Text-and-image in / text out, with automatic fallback across several free LLM APIs.
#
# Providers (all optional except that at least one must be configured):
#   gemini      Google Gemini (native API, the only one that takes audio/video/PDF)
#   groq        Groq            fast, generous daily limits, has a vision model
#   cerebras    Cerebras        very fast, ~1M tokens/day free
#   mistral     Mistral AI      good multilingual quality, vision, ~1 request/s free
#   nvidia      NVIDIA NIM      ~40 requests/min, big model catalogue (phone verification)
#   openrouter  OpenRouter      many ":free" models (50 req/day, 1000/day after a one-off $10 top-up)
#   cloudflare  Cloudflare Workers AI   10k "neurons"/day, needs an account id too
#   sambanova   SambaNova       20 req/min but only ~20/day
#   cohere      Cohere          Command A, strong multilingual, trial key: 20/min, 1000/month
#   zai         Z.ai / Zhipu    GLM "Flash" models are free (1 concurrent request), has a vision model
#   llm7        LLM7.io         no signup needed (opt-in, see below)
#
# Every provider is only used if its key is present in botkey.py. Order matters:
# the first provider that answers wins, the rest are fallbacks.
#
# ---------------------------------------------------------------------------
# botkey.py variables  (see also botkey.example.py)
# ---------------------------------------------------------------------------
# Each *_API_KEY may be one string or a list of strings (keys are rotated). For
# every provider, keys from DIFFERENT accounts/projects are what gives extra quota.
#
#   GEMINI_API_KEY      = ["AIza...", "AIza..."]     # https://aistudio.google.com/apikey
#   GROQ_API_KEY        = "gsk_..."                  # https://console.groq.com/keys 
#   MISTRAL_API_KEY     = "..."                      # https://console.mistral.ai/api-keys  (free "Experiment" plan)
#   NVIDIA_API_KEY      = "nvapi-..."                # https://build.nvidia.com  (profile -> API keys)
#   OPENROUTER_API_KEY  = "sk-or-..."                # https://openrouter.ai/keys
# Optional tuning (all have defaults):
#
#   PROVIDER_ORDER   = ["groq", "gemini", "cerebras"]   # order AND whitelist; names as in the list above
#                                                      # (a comma separated string also works)
#   ROTATE_PROVIDERS = True     # start each request on the next provider instead of always the first,
#                               # which spreads usage so no single free tier is drained first
#   <NAME>_MODELS        = ["model-a", "model-b"]       # replace a provider's text-model list, e.g. GROQ_MODELS
#   <NAME>_VISION_MODELS = ["model-v"]                  # replace a provider's image-capable model list
#
# Free-tier model names change often. If a model id stops existing the provider answers
# 404 and the client skips it for a few hours, but update the lists below (or override
# them from botkey.py with <NAME>_MODELS) when you notice it.
#
# IMPORTANT: Gemini quotas are per Google Cloud *project*, not per key. Keys that
# were created in the same project share one quota, so rotating between them
# gains nothing. Create each key in its own project (AI Studio: "Create API key
# in new project"). The same idea holds for the other providers: a second key from
# the same account usually shares the account's quota.
#
# Usage from the bot:
#   from llmclient import ask, Attachment, key_status, last_backend
#   reply = ask("hello")                 # -> str
#   last_backend()                       # -> "groq/llama-3.3-70b-versatile" (for this thread)
#   key_status()                         # snapshot of every provider's keys, for an owner-only command
#
# Exceptions (the Gemini* names are kept as aliases so existing `except` clauses keep working):
#   LLMUnavailable  / GeminiUnavailable   everything overloaded or out of quota; has .retry_after
#   LLMBlocked      / GeminiBlocked       the model refused (safety / policy)
#   LLMKeysRejected / GeminiKeysRejected  every key of every provider is invalid
#   ValueError                            unsupported attachment, or no configured provider can take it
#   RuntimeError                          every provider rejected the request itself (malformed)

import base64
import http.client
import json
import logging
import mimetypes
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import botkey

log = logging.getLogger("llmclient")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _cfg(name, default=None):
    return getattr(botkey, name, default)


def _as_list(value) -> list:
    """A string or an iterable of strings -> list of unique, stripped, non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    try:
        out = [v.strip() for v in value if isinstance(v, str) and v.strip()]
    except TypeError:
        return []
    return list(dict.fromkeys(out))


def _models_cfg(name: str, suffix: str, default: list) -> list:
    """botkey.<NAME>_MODELS / <NAME>_VISION_MODELS if set, else the built-in default."""
    override = _as_list(_cfg(f"{name.upper()}_{suffix}"))
    return override or list(default)


# ---------------------------------------------------------------------------
# Models per provider (best first). Edit freely, or override from botkey.py.
# ---------------------------------------------------------------------------

# Gemini. Tried in order. When the first is overloaded (503) or rate-limited (429) on
# every key, the next one usually still has capacity; they are separate capacity pools.
MODEL_CHAIN = [
    # Current generation Flash models
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash",

    # Current & previous generation Flash-Lite models
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

# Everything except Gemini speaks the OpenAI chat-completions dialect.
#   models / vision   text models, and the subset that accepts images
#   max_chars         largest prompt (+ system prompt) the free tier takes comfortably; bigger
#                     requests go to other providers first and are clipped only as a last resort
#   max_images        most images per request
#   scope             what a 429 limits: "model" (only that model) or "key" (the whole account)
#   reset             when daily quotas renew, if known: "utc" or "pacific"
_OPENAI_SPECS = {
    "groq": dict(
        url="https://api.groq.com/openai/v1/chat/completions",
        models=["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
                "meta-llama/llama-4-scout-17b-16e-instruct"],
        vision=["meta-llama/llama-4-scout-17b-16e-instruct"],
        max_chars=30_000, max_images=5, scope="model"),
    "mistral": dict(
        url="https://api.mistral.ai/v1/chat/completions",
        models=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
        vision=["mistral-small-latest", "mistral-medium-latest"],
        max_chars=200_000, max_images=8, scope="key"),
    "nvidia": dict(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        models=["meta/llama-3.3-70b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct",
                "meta/llama-3.1-70b-instruct"],
        vision=["meta/llama-3.2-90b-vision-instruct"],
        max_chars=100_000, max_images=1, scope="key"),
    "openrouter": dict(
        url="https://openrouter.ai/api/v1/chat/completions",
        models=["openrouter/free", "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemma-3-27b-it:free"],
        vision=["google/gemma-3-27b-it:free"],
        max_chars=100_000, max_images=4, scope="key", reset="utc",
        extra_body={"max_tokens": 4096},
        extra_headers={"HTTP-Referer": "https://github.com/", "X-Title": "discord-bot"}),
}

# Default provider order. First configured provider that answers wins.
DEFAULT_PROVIDER_ORDER = ["gemini", "groq", "cerebras", "mistral", "nvidia", "openrouter",
                          "cloudflare", "sambanova", "cohere", "zai", "llm7"]

# True: each request starts on the next provider (round robin among those that can serve it),
# so the free tiers are used evenly. False: always start at the top of the order and use the
# rest only as fallbacks. Overridable with botkey.ROTATE_PROVIDERS.
ROTATE_PROVIDERS = bool(_cfg("ROTATE_PROVIDERS", False))

MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_URL = MODELS_URL + "/{model}:generateContent"

# On the first request, ask the Gemini API which of MODEL_CHAIN actually exist (and
# support generateContent) and drop the rest, so a retired/misspelled model id
# never costs a round trip. If the listing fails, MODEL_CHAIN is used as-is.
VALIDATE_MODELS = True

# Soft guard against pathologically long callers. Gemini Flash accepts far more
# than this, but an unbounded prompt from a caller like the profiler eventually
# blows past HTTP body limits after base64 overhead, with a confusing 4xx.
MAX_PROMPT_CHARS = 200_000

USER_AGENT = "Mozilla/5.0 (compatible; llmclient/2.0; +discord-bot)"   # some CDNs reject urllib's default

# Tells the model it is talking through Discord and exactly how math is shown.
# Keep this in sync with latexrenderer.py: the "supported" list below is what
# matplotlib's mathtext can actually draw (checked against matplotlib 3.10).
SYSTEM_PROMPT = r"""You are a helpful assistant replying in a Discord chat. Reply in the same language the user writes in (Finnish if they write Finnish).

HOW MATH IS DISPLAYED
Discord cannot typeset LaTeX. A renderer on the bot turns math into images, but only math written inside delimiters:
- Inline math: $...$   (use for short expressions inside a sentence)
- Display math: $$...$$   (use for any longer, standalone, or important formula; put it on its own lines)
Everything outside the delimiters is shown as ordinary Discord Markdown (**bold**, *italic*, numbered lists, short headings are all fine).

RULES
1. Every formula, symbol, variable with an index or exponent, and number-with-math goes inside $...$ or $$...$$. Never write bare LaTeX like \frac{1}{2} or x^2 in normal text.
2. Never put math inside code blocks or backticks. Code blocks are only for real source code.
3. When the user asks for math, exercises or formulas "in LaTeX" ("latex-muodossa"), they want them typeset and readable in the chat, NOT a LaTeX source file. Write normal Markdown (numbered list, short intro) with the formulas in $...$ / $$...$$. Never output \documentclass, \usepackage, \begin{document}, \section, \title, \maketitle, itemize/enumerate, or theorem environments. Only if the user explicitly asks for the LaTeX source code / a .tex file, give it inside a ```latex code block.
4. Put words inside math in \text{...}, for example $x \ge 0 \text{ kun } n \to \infty$. Do not wrap whole sentences in $...$.
5. Use $ only for math. Write currency as "5 €" or "5 dollars", never "$5".
6. Multi-line derivations: ONE $$...$$ block, rows separated by \\ , alignment marks & are allowed. Do not use \begin{align}, \begin{equation} or \[ \] unless inside $$...$$.
7. Piecewise functions: use \begin{cases} a & \text{if } x>0 \\ b & \text{otherwise} \end{cases} inside $$...$$.
8. Keep each $$ block short (one idea, one to four rows). Prefer several small blocks over one huge one.

SUPPORTED MATH (use only these; unsupported commands fail and show as raw source)
Fractions and roots: \frac{a}{b}, \sqrt{x}, \sqrt[n]{x}, \binom{n}{k}
Big operators: \sum_{k=1}^{n}, \prod, \int_a^b, \oint, \lim_{x \to 0}, \sup, \inf, \max, \min, \limsup, \liminf
Functions: \sin \cos \tan \arctan \ln \log \exp
Greek letters: \alpha \beta \gamma \delta \varepsilon \epsilon \theta \lambda \mu \pi \sigma \phi \varphi \omega \Delta \Omega ...
Relations: = \neq \le \ge \leq \geq < > \approx \sim \equiv \in \notin \subset \subseteq \supset \supseteq \mid \to \Rightarrow \Leftrightarrow \iff \implies \mapsto
Logic and sets: \forall \exists \neg \wedge \vee \cup \cap \setminus \emptyset \mathbb{R} \mathbb{N} \mathbb{Z} \mathbb{Q} \mathbb{C}
Other: \infty \partial \nabla \cdot \times \pm \ldots \cdots \to, ^{ } and _{ } for powers and indices, |x|, \|x\|, \left( \right), \lfloor \rfloor, \lceil \rceil, \overline{x}, \hat{x}, \vec{v}, f'(x), \mathrm{d}x, \mathbf{x}, \mathcal{F}, \text{...}, \, \quad
DO NOT USE (not supported): \begin{matrix}/pmatrix/bmatrix/array/tabular, \xrightarrow, \underbrace, \overbrace, \boxed, \cancel, \color, \tag, \label, \newcommand, \usepackage, \begin{align} outside $$, tikz, \displaystyle. For a matrix, write its rows as separate lines of text instead.

GRAPHS OF FUNCTIONS, CURVES AND REGIONS
You cannot draw images, but the bot can draw graphs for you. Put a plot tag on its own line:
[[plot: f(x)]]    or    [[plot: f(x); g(x) from A to B]]
- Inside the tag use plain calculator syntax, NOT LaTeX: x^2, 2*x+1, sin(x), cos(x), tan(x), sqrt(x), abs(x), exp(x), ln(x), log10(x), pi, e.
- Graph of a function of x: write just the expression, e.g. [[plot: x^2 - 2*x]] (or "y = x^2 - 2*x").
- Curve that is not a function of x (circle, ellipse, hyperbola, any equation in x and y): write the equation with a single "=", e.g. [[plot: x^2 + y^2 = 1]], [[plot: x^2/4 + y^2 = 1]], [[plot: x*y = 1]], [[plot: (x-1)^2 + (y+2)^2 = 9]]. The variables are x and y only.
- Inequality (shaded region): write it with <, >, <= or >= (ASCII, not ≤ ≥), e.g. [[plot: x^2 + y^2 < 1]], [[plot: y > x^2]], [[plot: x + y <= 2; x > 0]]. The boundary is drawn dashed for < and >, solid for <= and >=; with several inequalities the shaded regions overlap, and the darker overlap is the solution of the system. Use this when asked to illustrate the solution set of an inequality or system of inequalities. Only one comparison per inequality (no a < x < b; write it as two: x > a; x < b).
- Separate up to 6 graphs with ;  The range "from A to B" is optional and applies to BOTH axes. For functions the default is -10 to 10, so choose a range that shows the interesting behaviour, e.g. [[plot: sin(x); cos(x) from -6.3 to 6.3]]. For curves in x and y, omit the range: the view fits automatically.
- Use a tag whenever the user asks to draw, plot or sketch a function or curve, or when a graph clearly helps an explanation (at most two per reply). Also state the function or equation in the text as math, e.g. "Yksikköympyrä $x^2 + y^2 = 1$:" followed by [[plot: x^2 + y^2 = 1]].
- Never draw graphs as ASCII art, and never put plot tags inside code blocks. (Function plots use [[plot: ...]]; the diagram tags below are for structures.)

DIAGRAMS: GRAPHS, AUTOMATA, TURING MACHINES, TREES
The bot can also draw graph-theory graphs, finite automata, Turing machines, pushdown automata and trees. Put a tag on its own line. The tag name is graph, tree, dfa, nfa, tm or pda; the content is a list of statements separated by newlines or ;
- Edges: "A -- B" (undirected), "A -> B" (directed), "A <-> B" (both ways), "A -> B : 5" (label/weight). For automata put the label between the dashes: "q0 -a-> q1", "q0 -a,b-> q0" (a self-loop is fine). Turing machine rule "read/write,move": "q0 -1/0,R-> q1" (blank is _).
- Other statements: "start: q0", "accept: q2, q3" (double circle), "nodes: x, y" (isolated nodes), "highlight: q1" (colours nodes, e.g. a path or the current state), "title: ...", "layout: circle|spring|lr|tb|tree" (optional; sensible default per tag).
- Turing machine tape snapshot: "tape: 1 0 1 1 _", "head: 2" (0-based cell under the head), "state: q1".
- Node names are single words (letters, digits, _): q0, q_accept, A, 3. Write epsilon as eps. No spaces inside names, no LaTeX, no $ in tags.
Examples:
[[dfa: start: q0; accept: q1; q0 -0-> q0; q0 -1-> q1; q1 -0,1-> q1]]
[[graph: A -- B : 3; B -- C : 1; A -- C : 7]]
[[tree: root -> L; root -> R; L -> LL; L -> LR]]
[[tm: start: q0; accept: qacc; q0 -1/1,R-> q0; q0 -_/_,L-> qacc]]
- Use a tag whenever the user asks to draw, show or illustrate such a structure, or when it clearly helps. At most two diagram tags per reply (plot tags count too). State what the diagram shows in the text, e.g. the language an automaton accepts.
- Never draw these as ASCII art, and never output Graphviz/DOT, Mermaid or TikZ code instead of a tag. Do not put diagram tags inside code blocks.

STYLE
Be concise and clear. Discord messages are limited, so avoid long preambles. For exercises, give a numbered list with each task stated precisely, and add hints or solutions only if asked.
"""

# Gemini accepts images, audio, video and PDF as inline data. The other providers
# take images only (and only some of their models).
SUPPORTED_PREFIXES = ("image/", "audio/", "video/", "application/pdf")


class Attachment:
    """One piece of non-text input: raw bytes plus its MIME type."""

    def __init__(self, data: bytes, mime_type: str, name: str = ""):
        self.data = data
        self.mime_type = mime_type
        self.name = name

    @classmethod
    def from_path(cls, path: str) -> "Attachment":
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            raise ValueError(f"Could not guess a MIME type for {path!r}")
        with open(path, "rb") as f:
            return cls(f.read(), mime_type, os.path.basename(path))

    @classmethod
    def from_bytes(cls, data: bytes, filename: str) -> "Attachment":
        mime_type, _ = mimetypes.guess_type(filename)
        return cls(data, mime_type or "application/octet-stream", filename)

    def to_part(self) -> dict:
        """Gemini's inline_data part."""
        return {
            "inline_data": {
                "mime_type": self.mime_type,
                "data": base64.b64encode(self.data).decode("ascii"),
            }
        }

    def to_image_url_part(self) -> dict:
        """OpenAI-style image part (data URI)."""
        b64 = base64.b64encode(self.data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{self.mime_type};base64,{b64}"}}


def build_system_prompt(guild_context: str = None) -> str:
    """The base system prompt, plus optional per-guild context (e.g. recent chat
    history of that server). Nothing global is ever modified, so different
    guilds can't leak into each other."""
    if guild_context:
        return SYSTEM_PROMPT + "\n\n" + guild_context
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Retry / rotation tuning
# ---------------------------------------------------------------------------
RETRY_STATUSES = {408, 425, 500, 502, 503, 504, 529}   # transient: retry on the same key
ATTEMPTS_PER_KEY = 2                    # tries per key for transient errors
REQUEST_TIMEOUT = 60                    # seconds for one HTTP request to Gemini (big media)
OPENAI_REQUEST_TIMEOUT = 40             # ... and to every other provider
TOTAL_DEADLINE = 120                    # seconds for one whole ask(), all providers, models and keys

DEFAULT_429_COOLDOWN = 30               # when the provider doesn't say how long to wait
DAILY_FALLBACK_COOLDOWN = 3600          # daily quota hit but reset time unknown: re-check hourly
MAX_COOLDOWN = 24 * 3600
DEAD_KEY_COOLDOWN = 3600                # invalid / revoked / out-of-credit key: re-check hourly
FORBIDDEN_COOLDOWN = 600                # 403 that isn't clearly a dead key
MODEL_MISSING_COOLDOWN = 6 * 3600       # 404: model id retired/unknown
MODEL_OVERLOAD_COOLDOWN = 15            # model kept returning 5xx: try others first

# True: start each request on the next key (spreads load, keeps every key under
# its per-minute limit). False: stay on one key until it hits a limit.
SPREAD_LOAD = True

BLOCK_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII",
                        "RECITATION", "IMAGE_SAFETY"}

# Phrases that mark a 429 as "daily quota used up" for the non-Gemini providers.
_DAILY_HINTS = ("per day", "daily", "neurons")


class LLMUnavailable(RuntimeError):
    """Everything is overloaded / rate-limited / unreachable (worth trying later).
    retry_after (seconds) is set when we know roughly when a key frees up.
    permanent is True when the request itself was refused as malformed or unsupported."""

    def __init__(self, msg, retry_after=None, permanent=False):
        super().__init__(msg)
        self.retry_after = retry_after
        self.permanent = permanent


class LLMBlocked(RuntimeError):
    """The model refused to answer (safety / policy). Retrying another model won't help."""


class LLMKeysRejected(RuntimeError):
    """Every configured API key of every provider is invalid or revoked. Needs a human."""


# Backwards compatible names.
GeminiUnavailable = LLMUnavailable
GeminiBlocked = LLMBlocked
GeminiKeysRejected = LLMKeysRejected

_ALL_KEYS = []          # every configured secret, for log redaction


def _redact(s: str) -> str:
    for k in _ALL_KEYS:
        s = s.replace(k, "***")
    return s


def _brief(text: str, limit: int = 200) -> str:
    return _redact(" ".join((text or "").split()))[:limit]


def _clip(prompt: str, limit: int) -> str:
    """Shorten an over-long prompt, keeping the start and (more importantly) the end:
    that's usually where the question is."""
    if limit <= 0 or len(prompt) <= limit:
        return prompt
    head = int(limit * 0.6)
    tail = limit - head
    return prompt[:head] + "\n[... truncated ...]\n" + prompt[-tail:]


# ---------------------------------------------------------------------------
# Key pool: all rotation state of ONE provider lives here, behind one lock.
# ---------------------------------------------------------------------------
class _KeyPool:
    def __init__(self, n: int):
        self.n = n
        self.lock = threading.Lock()
        self.cursor = 0
        self.key_until = [0.0] * n          # whole key unusable until (monotonic)
        self.key_reason = [""] * n
        self.pair_until = {}                # (key idx, model) -> monotonic time
        self.model_until = {}               # model overloaded -> monotonic time
        self.missing_until = {}             # model does not exist -> monotonic time

    def plan(self, model: str):
        """(ordered key indices usable for this model right now, soonest wait if none)."""
        now = time.monotonic()
        with self.lock:
            start = self.cursor
            if SPREAD_LOAD:
                self.cursor = (self.cursor + 1) % self.n
            order, soonest = [], None
            for i in range(self.n):
                idx = (start + i) % self.n
                until = max(self.key_until[idx], self.pair_until.get((idx, model), 0.0))
                if until <= now:
                    order.append(idx)
                else:
                    w = until - now
                    soonest = w if soonest is None else min(soonest, w)
            return order, soonest

    def stick(self, idx: int):
        if not SPREAD_LOAD:
            with self.lock:
                self.cursor = idx

    def cool_pair(self, idx: int, model: str, secs: float):
        with self.lock:
            self.pair_until[(idx, model)] = time.monotonic() + min(secs, MAX_COOLDOWN)

    def cool_key(self, idx: int, secs: float, reason: str):
        with self.lock:
            self.key_until[idx] = time.monotonic() + min(secs, MAX_COOLDOWN)
            self.key_reason[idx] = reason

    def cool_model(self, model: str, secs: float):
        with self.lock:
            self.model_until[model] = time.monotonic() + secs

    def cool_missing(self, model: str, secs: float):
        with self.lock:
            self.missing_until[model] = time.monotonic() + secs

    def model_cooling(self, model: str) -> bool:
        with self.lock:
            return self.model_until.get(model, 0.0) > time.monotonic()

    def model_missing(self, model: str) -> bool:
        with self.lock:
            return self.missing_until.get(model, 0.0) > time.monotonic()

    def soonest_wait(self):
        """Seconds until any key / (key, model) cooldown ends, or None."""
        now = time.monotonic()
        with self.lock:
            ends = [u - now for u in self.pair_until.values() if u > now]
            ends += [u - now for u in self.key_until if u > now]
            return min(ends) if ends else None

    def all_rejected(self) -> bool:
        now = time.monotonic()
        with self.lock:
            return all(self.key_until[i] > now and self.key_reason[i] == "invalid" for i in range(self.n))

    def snapshot(self, provider: str):
        now = time.monotonic()
        with self.lock:
            out = []
            for i in range(self.n):
                cooling = {m: round(u - now) for (k, m), u in self.pair_until.items() if k == i and u > now}
                state = "ok"
                if self.key_until[i] > now:
                    state = self.key_reason[i] or "cooling"
                out.append({"provider": provider, "key": f"#{i + 1}", "state": state,
                            "cooling_models": cooling})
            return out


# ---------------------------------------------------------------------------
# Error handling helpers
# ---------------------------------------------------------------------------
def _secs_until_pacific_midnight() -> float:
    """Gemini's daily quotas reset at midnight Pacific time."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/Los_Angeles"))
        nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (nxt - now).total_seconds() + 5
    except Exception:
        return 3600.0   # no tz database (e.g. Windows without tzdata): re-check hourly


def _secs_until_utc_midnight() -> float:
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (nxt - now).total_seconds() + 5


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)")
_RETRY_IN_RE = re.compile(r"(?:try again|retry|reset|resets|available)\s+in\s+((?:\d+(?:\.\d+)?\s*(?:ms|h|m|s)\s*)+)")


def _parse_duration(text: str):
    """'2m59.5s' / '7.5s' / '350ms' / '1h2m' -> seconds, or None."""
    total, found = 0.0, False
    for num, unit in _DURATION_PART.findall(text or ""):
        found = True
        n = float(num)
        total += n * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else None


def _quota_cooldown(p, text: str, headers) -> float:
    """How long to leave a (key, model) / key alone after a 429, from the provider's own hints."""
    delay, daily = None, False

    try:
        ra = headers.get("Retry-After") if headers is not None else None
        if ra:
            delay = float(ra)
    except (ValueError, TypeError, AttributeError):
        pass

    try:
        data = json.loads(text)
        if isinstance(data, list) and data:
            data = data[0]
        err = data.get("error", data) if isinstance(data, dict) else {}
        if isinstance(err, dict):
            for d in err.get("details", []) or []:
                t = d.get("@type", "")
                if t.endswith("RetryInfo") and d.get("retryDelay") and delay is None:
                    delay = _parse_duration(str(d["retryDelay"]))
                elif t.endswith("QuotaFailure"):
                    for v in d.get("violations", []) or []:
                        if "perday" in (str(v.get("quotaId", "")) + str(v.get("quotaMetric", ""))).lower():
                            daily = True
            if delay is None:
                m = re.search(r"retry in ([\d.]+)s", str(err.get("message", "")))
                if m:
                    delay = float(m.group(1))
            # OpenRouter: absolute reset time (ms since epoch) in the error metadata.
            hdrs = (err.get("metadata") or {}).get("headers") or {}
            reset = hdrs.get("X-RateLimit-Reset")
            if reset is not None and delay is None:
                d = float(reset) / 1000.0 - time.time()
                if d > 0:
                    delay = d
    except (ValueError, AttributeError, TypeError):
        pass

    low = re.sub(r"[-_]+", " ", text.lower())
    if delay is None:
        m = _RETRY_IN_RE.search(low)
        if m:
            delay = _parse_duration(m.group(1))
    if p.name != "gemini" and any(s in low for s in _DAILY_HINTS):
        daily = True

    if daily:
        if delay is not None and delay >= 120:
            secs = delay                         # the provider told us exactly
        else:
            secs = p.secs_until_reset() or delay or DAILY_FALLBACK_COOLDOWN
    else:
        secs = delay if delay is not None else DEFAULT_429_COOLDOWN
    return min(secs, MAX_COOLDOWN) + random.uniform(0.5, 1.5)


def _handle_http_error(p, idx: int, model: str, code: int, text: str, headers=None) -> str:
    """Apply cooldowns for an HTTP error. Returns what the caller should do:
    'rotate' (try the next key), 'transient' (retry same key), 'model_missing', 'skip'
    (this request/model can't be served here: move on, no cooldown)."""
    low = text.lower()
    pool = p.pool
    label = f"{p.name} key#{idx + 1}"
    if code == 429:
        secs = _quota_cooldown(p, text, headers)
        if p.quota_scope == "key":
            pool.cool_key(idx, secs, "rate-limited")
            log.warning("%s rate-limited, cooling the key %.0fs", label, secs)
        else:
            pool.cool_pair(idx, model, secs)
            log.warning("%s rate-limited on %s, cooling %.0fs", label, model, secs)
        return "rotate"
    if code == 402:
        pool.cool_key(idx, DEAD_KEY_COOLDOWN, "no-credit")
        log.warning("%s has no credit left (HTTP 402)", label)
        return "rotate"
    if code == 404:
        pool.cool_missing(model, MODEL_MISSING_COOLDOWN)
        log.warning("%s: model %s not found (404), skipping it for a while", p.name, model)
        return "model_missing"
    if code in RETRY_STATUSES:
        return "transient"
    if code == 401 or (code == 400 and any(s in low for s in ("api_key_invalid", "api key not valid", "api key expired"))):
        pool.cool_key(idx, DEAD_KEY_COOLDOWN, "invalid")
        log.error("%s was rejected as invalid (HTTP %d)", label, code)
        return "rotate"
    if code == 403:
        if any(s in low for s in ("leaked", "suspended", "disabled")):
            pool.cool_key(idx, DEAD_KEY_COOLDOWN, "invalid")
            log.error("%s looks revoked/suspended (HTTP 403)", label)
        else:
            pool.cool_pair(idx, model, FORBIDDEN_COOLDOWN)
            log.warning("%s forbidden on %s (HTTP 403)", label, model)
        return "rotate"
    # 400 / 413 / 422 / anything else: the request doesn't suit this model (too big, no
    # image support, bad parameter). Another model or provider may still take it.
    log.warning("%s: %s answered HTTP %d: %s", p.name, model, code, _brief(text, 160))
    return "skip"


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class _Provider:
    """One backend: its keys, its models, its quota state, and the HTTP dialect."""

    name = ""
    accepts_media = False        # audio / video / PDF
    quota_scope = "model"        # what a 429 limits: "model" or "key"
    reset = None                 # "utc" | "pacific" | None
    request_timeout = OPENAI_REQUEST_TIMEOUT
    max_chars = 100_000
    max_images = 0

    def __init__(self, keys: list, models: list, vision_models: list):
        self.keys = keys
        self.models = models
        self.vision_models = vision_models
        self.pool = _KeyPool(len(keys))

    # -- capabilities ------------------------------------------------------
    def secs_until_reset(self):
        if self.reset == "pacific":
            return _secs_until_pacific_midnight()
        if self.reset == "utc":
            return _secs_until_utc_midnight()
        return None

    def can_handle(self, attachments: list) -> bool:
        if not attachments:
            return True
        if any(not a.mime_type.startswith("image/") for a in attachments) and not self.accepts_media:
            return False
        return bool(self.vision_models) and len(attachments) <= self.max_images

    def candidates(self, need_vision: bool) -> list:
        """Models to try for this request, best first. Ones that returned 404 are left out;
        ones that were overloaded a moment ago go last rather than being skipped."""
        if need_vision:
            base = list(self.vision_models)
        else:
            base = list(self.models) + [m for m in self.vision_models if m not in self.models]
        base = [m for m in base if not self.pool.model_missing(m)]
        return [m for m in base if not self.pool.model_cooling(m)] + [m for m in base if self.pool.model_cooling(m)]

    # -- HTTP dialect (overridden) ----------------------------------------
    def url(self, model: str, idx: int) -> str:
        raise NotImplementedError

    def headers(self, idx: int) -> dict:
        raise NotImplementedError

    def build_body(self, model: str, prompt: str, attachments: list, system_prompt: str) -> bytes:
        raise NotImplementedError

    def parse(self, payload: dict, model: str) -> str:
        raise NotImplementedError

    def _post(self, model: str, idx: int, body: bytes, timeout: float) -> dict:
        req = urllib.request.Request(self.url(model, idx), data=body, headers=self.headers(idx), method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    # -- one model, all keys ----------------------------------------------
    def call(self, prompt: str, attachments: list, model: str, system_prompt: str, deadline: float) -> str:
        """One model of this provider. Tries every key that isn't cooling down; rotates on
        429 / rejected keys, retries transient errors on the same key with a short jittered
        backoff.

        Raises LLMUnavailable if no key could answer, LLMBlocked if the model refused the
        content."""
        label = f"{self.name}/{model}"
        body = self.build_body(model, prompt, attachments, system_prompt)

        order, soonest = self.pool.plan(model)
        if not order:
            raise LLMUnavailable(
                f"{label}: every key is cooling down" + (f" (next frees in {soonest:.0f}s)" if soonest else ""),
                retry_after=soonest,
            )

        last_err = ""
        for idx in order:
            for attempt in range(1, ATTEMPTS_PER_KEY + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 1:
                    raise LLMUnavailable(f"{label}: out of time ({last_err})")
                try:
                    payload = self._post(model, idx, body, min(self.request_timeout, remaining))
                except urllib.error.HTTPError as e:
                    text = e.read().decode("utf-8", "replace")[:2000]
                    last_err = f"key#{idx + 1} HTTP {e.code}: {_brief(text, 120)}"
                    kind = _handle_http_error(self, idx, model, e.code, text, e.headers)
                    if kind == "rotate":
                        break                               # next key, no sleeping
                    if kind == "model_missing":
                        raise LLMUnavailable(f"{label}: not found")
                    if kind == "skip":
                        raise LLMUnavailable(f"{label}: {last_err}", permanent=True)
                    # "transient": fall through to backoff + retry on this key
                except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as e:
                    # URLError/timeouts, connection reset mid-body, truncated or non-JSON body
                    last_err = f"key#{idx + 1} {type(e).__name__}: {_brief(str(e), 120)}"
                else:
                    self.pool.stick(idx)
                    return self.parse(payload, model)
                if attempt < ATTEMPTS_PER_KEY:
                    pause = random.uniform(0.7, 1.3) * 1.5 * attempt
                    time.sleep(max(0.0, min(pause, deadline - time.monotonic() - 1)))
            else:
                # Transient errors on every attempt: the model itself is struggling and
                # another key won't change that. Let the next model go first for a while.
                self.pool.cool_model(model, MODEL_OVERLOAD_COOLDOWN)
                raise LLMUnavailable(f"{label}: {last_err}")

        raise LLMUnavailable(f"{label}: no key could answer ({last_err})", retry_after=self.pool.soonest_wait())


class _GeminiProvider(_Provider):
    name = "gemini"
    accepts_media = True
    quota_scope = "model"
    reset = "pacific"
    request_timeout = REQUEST_TIMEOUT
    max_chars = MAX_PROMPT_CHARS
    max_images = 10_000

    def __init__(self, keys: list, models: list):
        super().__init__(keys, models, models)
        self._chain_lock = threading.Lock()
        self._chain = None

    def _list_models(self) -> set:
        """Names of models that support generateContent, using the first key that works."""
        last = None
        for key in self.keys:
            names, token = set(), None
            try:
                while True:
                    url = MODELS_URL + "?pageSize=1000" + (f"&pageToken={token}" if token else "")
                    req = urllib.request.Request(url, headers={"x-goog-api-key": key, "User-Agent": USER_AGENT})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = json.load(resp)
                    for m in data.get("models", []):
                        if "generateContent" in (m.get("supportedGenerationMethods") or []):
                            names.add(m.get("name", "").removeprefix("models/"))
                    token = data.get("nextPageToken")
                    if not token:
                        return names
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
                last = e
        raise RuntimeError(f"model listing failed: {last}")

    def _validated(self) -> list:
        if self._chain is not None:
            return self._chain
        with self._chain_lock:
            if self._chain is None:
                chain = list(self.models)
                if VALIDATE_MODELS:
                    try:
                        available = self._list_models()
                        kept = [m for m in self.models if m in available]
                        dropped = [m for m in self.models if m not in available]
                        if dropped:
                            log.warning("Gemini models not offered by the API, dropped: %s", ", ".join(dropped))
                        if kept:
                            chain = kept
                        else:
                            log.warning("none of the Gemini models found in the model list; using them unchanged")
                    except Exception as e:
                        log.warning("could not validate the Gemini model list (%s); using it unchanged", _brief(str(e)))
                self._chain = chain
        return self._chain

    def candidates(self, need_vision: bool) -> list:
        base = [m for m in self._validated() if not self.pool.model_missing(m)]
        return [m for m in base if not self.pool.model_cooling(m)] + [m for m in base if self.pool.model_cooling(m)]

    def url(self, model, idx):
        return GEMINI_URL.format(model=model)

    def headers(self, idx):
        return {"Content-Type": "application/json", "x-goog-api-key": self.keys[idx], "User-Agent": USER_AGENT}

    def build_body(self, model, prompt, attachments, system_prompt):
        parts = [a.to_part() for a in attachments]
        if prompt:
            parts.append({"text": prompt})
        return json.dumps({
            "systemInstruction": {"parts": [{"text": system_prompt or SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
        }).encode("utf-8")

    def parse(self, payload, model):
        candidates = payload.get("candidates") or []
        if not candidates:
            reason = (payload.get("promptFeedback") or {}).get("blockReason")
            if reason:
                raise LLMBlocked(f"Gemini blocked the request ({reason}).")
            raise LLMUnavailable(f"gemini/{model}: response had no candidates")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought")).strip()
        if not text:
            reason = candidate.get("finishReason")
            if reason in BLOCK_FINISH_REASONS:
                raise LLMBlocked(f"Gemini declined to answer ({reason}).")
            # MAX_TOKENS spent on thinking, OTHER, ...: another model may do better.
            raise LLMUnavailable(f"gemini/{model}: empty response (finishReason={reason})")
        return text


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


class _OpenAIProvider(_Provider):
    """Any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, name, keys, models, vision_models, spec, accounts=None):
        super().__init__(keys, models, vision_models)
        self.name = name
        self._url = spec["url"]
        self.max_chars = spec.get("max_chars", 100_000)
        self.max_images = spec.get("max_images", 0)
        self.quota_scope = spec.get("scope", "model")
        self.reset = spec.get("reset")
        self.extra_body = dict(spec.get("extra_body") or {})
        self.extra_headers = dict(spec.get("extra_headers") or {})
        self.accounts = accounts or []       # per-key account ids (Cloudflare)

    def url(self, model, idx):
        if self.accounts:
            return self._url.format(account=self.accounts[idx])
        return self._url

    def headers(self, idx):
        h = {"Content-Type": "application/json", "Accept": "application/json",
             "Authorization": f"Bearer {self.keys[idx]}", "User-Agent": USER_AGENT}
        h.update(self.extra_headers)
        return h

    def build_body(self, model, prompt, attachments, system_prompt):
        if attachments:
            content = [a.to_image_url_part() for a in attachments]
            content.append({"type": "text", "text": prompt or "Describe the image."})
        else:
            content = prompt or "(empty message)"     # plain string: the most widely accepted form
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        body.update(self.extra_body)
        return json.dumps(body).encode("utf-8")

    def parse(self, payload, model):
        choices = payload.get("choices") or []
        if not choices:
            err = payload.get("error")
            detail = _brief(json.dumps(err, ensure_ascii=False)) if err else "no choices"
            raise LLMUnavailable(f"{self.name}/{model}: {detail}")
        choice = choices[0]
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict) and p.get("type") in (None, "text"))
        text = _THINK_RE.sub("", content or "").strip()
        if not text:
            reason = choice.get("finish_reason")
            if reason == "content_filter":
                raise LLMBlocked("The model declined to answer (content_filter).")
            raise LLMUnavailable(f"{self.name}/{model}: empty response (finish_reason={reason})")
        return text


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
def _cloudflare_pairs():
    """[(token, account_id)], unique. CLOUDFLARE_ACCOUNT_ID is one id or a list aligned with the keys."""
    keys = _as_list_keep(_cfg("CLOUDFLARE_API_KEY"))
    ids = _as_list_keep(_cfg("CLOUDFLARE_ACCOUNT_ID"))
    if not keys:
        return []
    if not ids:
        log.warning("CLOUDFLARE_API_KEY is set but CLOUDFLARE_ACCOUNT_ID is missing; Cloudflare disabled")
        return []
    if len(ids) == 1:
        ids = ids * len(keys)
    if len(ids) != len(keys):
        log.warning("CLOUDFLARE_ACCOUNT_ID must be one id or one per key; Cloudflare disabled")
        return []
    return list(dict.fromkeys(zip(keys, ids)))


def _as_list_keep(value) -> list:
    """Like _as_list but keeps duplicates and order (needed to align keys with account ids)."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    try:
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]
    except TypeError:
        return []


def _build_providers() -> list:
    built = {}

    gkeys = _as_list(_cfg("GEMINI_API_KEY"))
    if gkeys:
        built["gemini"] = _GeminiProvider(gkeys, _models_cfg("GEMINI", "MODELS", MODEL_CHAIN))

    for name, spec in _OPENAI_SPECS.items():
        accounts = None
        if name == "cloudflare":
            pairs = _cloudflare_pairs()
            keys = [k for k, _ in pairs]
            accounts = [a for _, a in pairs]
        elif name == "llm7":
            keys = _as_list(_cfg("LLM7_API_KEY"))
            if not keys and _cfg("LLM7_ENABLED", False):
                keys = ["unused"]              # LLM7 accepts any token when you have none
        else:
            keys = _as_list(_cfg(f"{name.upper()}_API_KEY"))
        if not keys:
            continue
        models = _models_cfg(name, "MODELS", spec["models"])
        vision = _models_cfg(name, "VISION_MODELS", spec["vision"])
        built[name] = _OpenAIProvider(name, keys, models, vision, spec, accounts)

    order = _cfg("PROVIDER_ORDER") or DEFAULT_PROVIDER_ORDER
    if isinstance(order, str):
        order = [s.strip() for s in order.split(",")]
    order = [str(s).strip().lower() for s in order if str(s).strip()]

    unknown = [n for n in order if n not in DEFAULT_PROVIDER_ORDER]
    if unknown:
        log.warning("PROVIDER_ORDER names unknown providers: %s", ", ".join(unknown))
    left_out = [n for n in built if n not in order]
    if left_out:
        log.warning("providers with keys but not listed in PROVIDER_ORDER (disabled): %s", ", ".join(left_out))

    providers = [built[n] for n in dict.fromkeys(order) if n in built]
    if not providers:
        raise RuntimeError(
            "No LLM provider is configured. Set at least one of GEMINI_API_KEY, GROQ_API_KEY, "
            "CEREBRAS_API_KEY, MISTRAL_API_KEY, NVIDIA_API_KEY, OPENROUTER_API_KEY, "
            "CLOUDFLARE_API_KEY (+ CLOUDFLARE_ACCOUNT_ID), SAMBANOVA_API_KEY, COHERE_API_KEY, "
            "ZAI_API_KEY or LLM7_ENABLED in botkey.py")

    for p in providers:
        _ALL_KEYS.extend(k for k in p.keys if len(k) >= 8)
    for p in providers:
        for acct in getattr(p, "accounts", []):
            if len(acct) >= 8:
                _ALL_KEYS.append(acct)
    return providers


_providers = _build_providers()
_rotate_lock = threading.Lock()
_rotate_cursor = 0
_tls = threading.local()


def providers_configured() -> list:
    """Names of the active providers, in order."""
    return [p.name for p in _providers]


def key_status() -> list:
    """Rotation state per key, without exposing the keys:
    [{'provider': 'groq', 'key': '#1', 'state': 'ok'|'invalid'|'rate-limited'|'no-credit', 'cooling_models': {model: secs}}]"""
    rows = []
    for p in _providers:
        rows.extend(p.pool.snapshot(p.name))
    return rows


def last_backend():
    """'provider/model' that answered the last ask() made by the calling thread, or None."""
    return getattr(_tls, "backend", None)


# ---------------------------------------------------------------------------
# The provider chain
# ---------------------------------------------------------------------------
def _generate(prompt: str, attachments: list, system_prompt: str) -> str:
    """Walk providers (and each one's models) until something answers."""
    global _rotate_cursor
    deadline = time.monotonic() + TOTAL_DEADLINE
    need_vision = bool(attachments)

    capable = [p for p in _providers if p.can_handle(attachments)]
    if not capable:
        kinds = sorted({a.mime_type for a in attachments})
        raise ValueError(
            f"None of the configured providers ({', '.join(providers_configured())}) can take this "
            f"attachment ({', '.join(kinds)}, {len(attachments)} file(s)). "
            "Audio, video and PDF need GEMINI_API_KEY; images need a provider with a vision model.")

    size = len(prompt) + len(system_prompt)
    fits = [p for p in capable if size <= p.max_chars]
    too_big = [p for p in capable if size > p.max_chars]
    if ROTATE_PROVIDERS and len(fits) > 1:
        with _rotate_lock:
            shift = _rotate_cursor % len(fits)
            _rotate_cursor += 1
        fits = fits[shift:] + fits[:shift]
    # Providers whose free tier is too small for this prompt come last, with the prompt clipped.
    plan = [(p, False) for p in fits] + [(p, True) for p in too_big]

    errors, waits, perms = [], [], []
    for p, clip in plan:
        if time.monotonic() >= deadline:
            errors.append("deadline reached")
            perms.append(False)
            break
        pr = _clip(prompt, p.max_chars - len(system_prompt)) if clip else prompt
        for model in p.candidates(need_vision):
            if time.monotonic() >= deadline:
                break
            try:
                text = p.call(pr, attachments, model, system_prompt, deadline)
            except LLMUnavailable as e:
                errors.append(str(e))
                perms.append(e.permanent)
                if e.retry_after is not None:
                    waits.append(e.retry_after)
                continue
            _tls.backend = f"{p.name}/{model}"
            log.info("answered by %s", _tls.backend)
            return text

    if all(p.pool.all_rejected() for p in _providers):
        raise LLMKeysRejected("All API keys of all providers were rejected as invalid. Check botkey.py.")
    if errors and all(perms):
        raise RuntimeError("Every provider rejected the request: " + " | ".join(errors))
    log.warning("all providers unavailable: %s", " | ".join(errors))
    soonest = [w for w in (p.pool.soonest_wait() for p in _providers) if w is not None]
    raise LLMUnavailable(" | ".join(errors),
                         retry_after=min(waits) if waits else (min(soonest) if soonest else None))


def ask(prompt: str, attachments=None, guild_context: str = None) -> str:
    """
    Send a prompt, optionally with attachments (list of Attachment), and
    return the reply as a string.

    Tries the configured providers in order (see PROVIDER_ORDER), each with its
    model chain, rotating API keys on 429 / rejected keys. guild_context
    (optional) is extra text appended to the system prompt for this one call,
    used by the bot to give each Discord server its own memory.

    Images work with every provider that has a vision model; audio, video and
    PDF need Gemini. Which backend answered is available from last_backend().

    Raises LLMUnavailable (retry later), LLMBlocked (content refused),
    LLMKeysRejected (all keys invalid), ValueError (unsupported attachment) or
    RuntimeError (malformed request).
    """
    attachments = attachments or []
    system_prompt = build_system_prompt(guild_context)

    for a in attachments:
        if not a.mime_type.startswith(SUPPORTED_PREFIXES):
            raise ValueError(f"Unsupported attachment type: {a.mime_type}")

    prompt = _clip(prompt or "", MAX_PROMPT_CHARS)

    try:
        return _generate(prompt, attachments, system_prompt)
    except LLMUnavailable as e:
        # Full per-model detail is in the log; keep the chat message short.
        hint = f" Try again in about {int(e.retry_after) + 1} s." if e.retry_after else " Please try again in a minute."
        raise LLMUnavailable("The AI is overloaded or out of quota right now." + hint,
                             retry_after=e.retry_after) from None


def main():
    """Interactive loop. Prefix a line with `file:<path>` to attach something.
    Type `status` to see the key rotation state."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    print("LLM test. Attach with: file:/path/to/img.png what is this?")
    for p in _providers:
        print(f"  {p.name}: {len(p.keys)} key(s), {len(p.models)} model(s)")
    print("Type 'status' for key state, 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bye.")
            break
        if user_input.lower() == "status":
            for row in key_status():
                print(f"  {row['provider']} key {row['key']}: {row['state']} {row['cooling_models'] or ''}")
            print()
            continue

        attachments = []
        while user_input.startswith("file:"):
            token, _, rest = user_input[5:].partition(" ")
            try:
                attachments.append(Attachment.from_path(os.path.expanduser(token)))
            except (OSError, ValueError) as e:
                print(f"[error] {e}\n")
                attachments = None
                break
            user_input = rest.strip()

        if attachments is None:
            continue

        try:
            reply = ask(user_input, attachments)
            print(f"Bot ({last_backend()}): {reply}\n")
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
