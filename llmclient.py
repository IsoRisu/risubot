# llmclient.py
# Text-and-image in / text out, with automatic fallback across several free LLM APIs.
#
# Providers (all optional except that at least one must be configured):
#   gemini      Google Gemini (native API, the only one that takes audio/video/PDF)
#   groq        Groq            fast, generous daily limits, has a vision model
#   mistral     Mistral AI      good multilingual quality, vision, ~1 request/s free
#   nvidia      NVIDIA NIM      ~40 requests/min, big model catalogue (phone verification)
#   openrouter  OpenRouter      many ":free" models (50 req/day, 1000/day after a one-off $10 top-up)
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
#   PROVIDER_ORDER   = ["groq", "gemini", "mistral"]   # order AND whitelist; names as in the list above
#                                                      # (a comma separated string also works)
#   ROTATE_PROVIDERS = True     # start each request on the next provider instead of always the first,
#                               # which spreads usage so no single free tier is drained first
#   <NAME>_MODELS        = ["model-a", "model-b"]       # replace a provider's text-model list, e.g. GROQ_MODELS
#   <NAME>_VISION_MODELS = ["model-v"]                  # replace a provider's image-capable model list
#
#   LOCAL_MODEL          = "qwen3-4b-safe"              # optional local model for Ollama tool loops
#   LOCAL_URL            = "http://localhost:11434/v1/chat/completions"
#   LOCAL_TIMEOUT        = 1800    # seconds per HTTP call to Ollama (a 4B on CPU is slow)
#   LOCAL_DEADLINE       = 3600    # seconds per tool round (all retries included)
#   LOCAL_MAX_CHARS      = 12000   # ~4096 ctx; raise together with num_ctx in the Modelfile
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
#
# Text / source-code attachments (.py .js .json .md .txt ...) work with every provider: ask()
# pastes their content into the prompt. In the bot, use is_supported_file(name, content_type, size)
# to decide whether to download an attachment, and Attachment.from_bytes(data, filename) to wrap it.
#   RuntimeError                          every provider rejected the request itself (malformed)
from __future__ import annotations

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
    "local": dict(
        url="http://localhost:11434/v1/chat/completions",
        models=["qwen3-4b-safe"],
        vision=[],
        max_chars=12_000, max_images=0, scope="model"),   # ~4096 ctx; raise together with num_ctx
}

# Default provider order. First configured provider that answers wins.
DEFAULT_PROVIDER_ORDER = ["gemini", "groq", "mistral", "nvidia", "openrouter"]

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
SYSTEM_PROMPT = r"""You are a Discord chat assistant. Reply in the user's language (Finnish if they write Finnish). Be concise; no long preambles. For exercises: numbered list, tasks stated precisely, hints/solutions only if asked.

MATH
Discord can't render LaTeX; the bot renders only math inside delimiters: $...$ inline, $$...$$ display (own lines, for longer/important formulas). Everything else is Discord Markdown.
Rules:
1. All formulas, symbols, indexed/exponent variables, and numbers-with-math go in $...$ or $$...$$. Never bare LaTeX in text.
2. Never put math in code blocks/backticks; code blocks are for real source code only.
3. If the user asks for math "in LaTeX" (latex-muodossa), they want typeset, readable output, not source: use Markdown (numbered list, short intro) with $/$$ formulas. Never output \documentclass, \usepackage, \begin{document}, \section, \title, \maketitle, itemize/enumerate, or theorem environments. Only if explicitly asked for LaTeX source/.tex, give it in a ```latex block.
4. Words inside math go in \text{...}, e.g. $x \ge 0 \text{ kun } n \to \infty$. Don't wrap whole sentences in $.
5. $ is for math only. Write currency as "5 €" / "5 dollars".
6. Multi-line derivations: ONE $$ block, rows separated by \\, & alignment allowed. No \begin{align}, \begin{equation}, \[ \] outside $$.
7. Piecewise: \begin{cases} a & \text{if } x>0 \\ b & \text{otherwise} \end{cases} inside $$.
8. Keep each $$ block to one idea, 1-4 rows; prefer several small blocks.

SUPPORTED (others fail and show raw)
\frac \sqrt[n] \binom \sum \prod \int \oint \lim \sup \inf \max \min \limsup \liminf \sin \cos \tan \arctan \ln \log \exp; Greek letters (\alpha \beta \gamma \delta \varepsilon \epsilon \theta \lambda \mu \pi \sigma \phi \varphi \omega \Delta \Omega ...); = \neq \le \ge \leq \geq < > \approx \sim \equiv \in \notin \subset \subseteq \supset \supseteq \mid \to \Rightarrow \Leftrightarrow \iff \implies \mapsto \forall \exists \neg \wedge \vee \cup \cap \setminus \emptyset \mathbb{R,N,Z,Q,C} \infty \partial \nabla \cdot \times \pm \ldots \cdots ^{} _{} |x| \|x\| \left( \right) \lfloor \rfloor \lceil \rceil \overline \hat \vec f'(x) \mathrm{d}x \mathbf \mathcal \text \, \quad
NOT supported: matrix/pmatrix/bmatrix/array/tabular, \xrightarrow \underbrace \overbrace \boxed \cancel \color \tag \label \newcommand \usepackage, align outside $$, tikz, \displaystyle. For matrices, write rows as separate text lines.

PLOTS
You can't draw images; the bot draws from a tag on its own line: [[plot: f(x)]] or [[plot: f(x); g(x) from A to B]]
- Plain calculator syntax, not LaTeX: x^2, 2*x+1, sin cos tan sqrt abs exp ln log10 pi e.
- Function of x: just the expression, e.g. [[plot: x^2 - 2*x]] (or "y = ...").
- Other curves: equation with a single "=", variables x,y only: [[plot: x^2 + y^2 = 1]], [[plot: x*y = 1]], [[plot: (x-1)^2 + (y+2)^2 = 9]].
- Inequalities (shaded): use <, >, <=, >= (ASCII): [[plot: y > x^2]], [[plot: x + y <= 2; x > 0]]. Dashed boundary for < >, solid for <= >=; overlap of several is the solution. One comparison per inequality (a<x<b → x > a; x < b).
- Up to 6 graphs separated by ;. "from A to B" optional, applies to both axes; function default is -10 to 10, so pick a range showing the interesting behavior, e.g. [[plot: sin(x); cos(x) from -6.3 to 6.3]]. Omit range for x-y curves (auto-fit).
- Use when the user asks to draw/plot/sketch, or when it clearly helps (max two per reply). Also state the function in text as math, e.g. "Yksikköympyrä $x^2 + y^2 = 1$:" then the tag.
- Never ASCII art; never tags inside code blocks.

DIAGRAMS
Tags: graph, tree, dfa, nfa, tm, pda; content is statements separated by newlines or ;
- Edges: "A -- B" undirected, "A -> B" directed, "A <-> B" both, "A -> B : 5" labeled/weighted. Automata: "q0 -a-> q1", "q0 -a,b-> q0" (self-loops ok). TM rule "read/write,move": "q0 -1/0,R-> q1" (blank is _).
- Statements: "start: q0", "accept: q2, q3", "nodes: x, y", "highlight: q1", "title: ...", "layout: circle|spring|lr|tb|tree" (optional).
- TM snapshot: "tape: 1 0 1 1 _", "head: 2" (0-based), "state: q1".
- Node names: single words (letters, digits, _). Epsilon = eps. No spaces, LaTeX, or $ in tags.
Examples:
[[dfa: start: q0; accept: q1; q0 -0-> q0; q0 -1-> q1; q1 -0,1-> q1]]
[[graph: A -- B : 3; B -- C : 1; A -- C : 7]]
[[tree: root -> L; root -> R; L -> LL; L -> LR]]
[[tm: start: q0; accept: qacc; q0 -1/1,R-> q0; q0 -_/_,L-> qacc]]
- Use when asked to draw/show/illustrate such structures or when it clearly helps. Max two diagram tags per reply (plots count). State in text what the diagram shows (e.g. the accepted language).
- Never ASCII art, Graphviz/DOT, Mermaid, or TikZ; no tags in code blocks.

CREATIVE WRITING
You're also a skilled fiction writer for a literature/storytelling/roleplay server. Write original fiction in all genres.
1. Explore dark, mature, intense themes where permitted: fictional violence, gore, death, profanity, morally complex characters. Don't sanitize scenes meant to be brutal, frightening, or tragic.
2. Respect the author's premise, characters, worldbuilding, and direction; keep continuity. When asked to write, deliver the actual story, not just ideas.
3. Follow the language rule and requested word counts/format/style. Avoid disclaimers, commentary, moralizing."""

# Gemini accepts images, audio, video and PDF as inline data. The other providers
# take images only (and only some of their models).
SUPPORTED_PREFIXES = ("image/", "audio/", "video/", "application/pdf")

# Text and source-code files are not sent as attachments at all: ask() decodes them and
# pastes their content into the prompt, so EVERY provider can read them.
MAX_TEXT_FILE_BYTES = 1_000_000      # larger files are refused (use is_supported_file() in the bot)
MAX_TEXT_FILE_CHARS = 150_000        # content beyond this is clipped (start + end kept)

TEXT_EXTENSIONS = frozenset({
    "txt", "md", "markdown", "rst", "log", "csv", "tsv", "json", "jsonl", "ndjson", "xml", "yaml", "yml",
    "toml", "ini", "cfg", "conf", "properties", "env", "gitignore", "lock", "tex", "bib", "srt", "vtt",
    "py", "pyw", "pyi", "ipynb", "js", "mjs", "cjs", "jsx", "ts", "tsx", "vue", "svelte", "html", "htm",
    "css", "scss", "sass", "less", "java", "kt", "kts", "scala", "groovy", "gradle", "c", "h", "cc", "cpp",
    "cxx", "hpp", "hh", "cs", "go", "rs", "rb", "php", "swift", "m", "mm", "dart", "lua", "pl", "pm", "r",
    "jl", "hs", "ml", "clj", "ex", "exs", "erl", "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd", "sql",
    "graphql", "proto", "tf", "asm", "s", "v", "vhd", "vhdl", "cmake", "mk", "diff", "patch",
})
TEXT_FILENAMES = frozenset({"dockerfile", "makefile", "readme", "license", "gemfile", "rakefile",
                            "procfile", "requirements.txt", ".gitignore", ".env.example"})
_TEXT_MIMES = frozenset({
    "application/json", "application/xml", "application/javascript", "application/x-javascript",
    "application/typescript", "application/x-sh", "application/x-python-code", "application/x-yaml",
    "application/yaml", "application/toml", "application/sql", "application/x-httpd-php",
    "application/x-tex", "application/x-latex", "application/x-ipynb+json", "application/x-csh",
})


def _ext(filename: str) -> str:
    name = os.path.basename(filename or "").lower()
    return name.rsplit(".", 1)[-1] if "." in name else ""


def is_text_filename(filename: str) -> bool:
    """True for source code / plain-text files, judged by name only."""
    name = os.path.basename(filename or "").lower()
    return name in TEXT_FILENAMES or _ext(name) in TEXT_EXTENSIONS


def _guess_mime(filename: str) -> str:
    """MIME type for a filename. Source code is always text/plain, because mimetypes
    returns None or odd values (application/x-sh, application/octet-stream ...) for it."""
    if is_text_filename(filename):
        return "text/plain"
    mime_type, _ = mimetypes.guess_type(filename or "")
    return mime_type or "application/octet-stream"


def is_supported_file(filename: str, mime_type: str = None, size: int = None) -> bool:
    """For the Discord bot: should this attachment be downloaded and passed to ask()?
    Replaces checks like `content_type.startswith("image/")` and the size limit."""
    mime_type = (mime_type or "").split(";")[0].strip().lower()
    if is_text_filename(filename) or mime_type.startswith("text/") or mime_type in _TEXT_MIMES:
        return size is None or size <= MAX_TEXT_FILE_BYTES
    return mime_type.startswith(SUPPORTED_PREFIXES) or _guess_mime(filename).startswith(SUPPORTED_PREFIXES)


class Attachment:
    """One piece of input: raw bytes plus its MIME type. Text/code files are allowed."""

    def __init__(self, data: bytes, mime_type: str, name: str = ""):
        self.data = data
        self.mime_type = (mime_type or "application/octet-stream").split(";")[0].strip().lower()
        self.name = name
        # Discord reports many code files as application/octet-stream or odd types:
        # trust the filename when it says "source code".
        if is_text_filename(name) and not self.mime_type.startswith(SUPPORTED_PREFIXES):
            self.mime_type = "text/plain"

    @property
    def is_text(self) -> bool:
        return self.mime_type.startswith("text/") or self.mime_type in _TEXT_MIMES

    @classmethod
    def from_path(cls, path: str) -> "Attachment":
        mime_type = _guess_mime(path)
        with open(path, "rb") as f:
            return cls(f.read(), mime_type, os.path.basename(path))

    @classmethod
    def from_bytes(cls, data: bytes, filename: str, mime_type: str = None) -> "Attachment":
        return cls(data, mime_type or _guess_mime(filename), filename)

    def decode_text(self) -> str:
        """The file as text. Raises ValueError for binary data or oversized files."""
        if len(self.data) > MAX_TEXT_FILE_BYTES:
            raise ValueError(f"{self.name or 'file'} is too large ({len(self.data) // 1000} kB, "
                             f"limit {MAX_TEXT_FILE_BYTES // 1000} kB).")
        if b"\x00" in self.data[:8192]:
            raise ValueError(f"{self.name or 'file'} looks like a binary file, not text.")
        for enc in ("utf-8-sig", "utf-16") if self.data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
            try:
                return self.data.decode(enc)
            except UnicodeDecodeError:
                pass
        try:
            return self.data.decode("cp1252")
        except UnicodeDecodeError:
            return self.data.decode("latin-1")

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


def _inline_text_files(prompt: str, text_files: list) -> str:
    """Put the content of text/code attachments into the prompt, files first and the
    user's own message LAST (when a prompt is too long, _clip keeps the end)."""
    blocks = []
    for a in text_files:
        text = _clip(a.decode_text().replace("\r\n", "\n"), MAX_TEXT_FILE_CHARS)
        blocks.append(f"=== BEGIN FILE: {a.name or 'file'} ===\n{text}\n=== END FILE: {a.name or 'file'} ===")
    if not blocks:
        return prompt
    question = (prompt or "").strip() or "Review the attached file(s): explain what they do and point out any problems."
    return "The user attached these file(s):\n\n" + "\n\n".join(blocks) + "\n\nUser's message:\n" + question


def build_system_prompt(guild_context: str = None, system_prompt: str = None) -> str:
    """The base system prompt (defaulting to SYSTEM_PROMPT or overridden by system_prompt),
    plus optional per-guild context (e.g. recent chat history of that server).
    Nothing global is ever modified, so different guilds can't leak into each other."""
    base = SYSTEM_PROMPT if system_prompt is None else system_prompt
    if guild_context:
        return base + "\n\n" + guild_context
    return base


# ---------------------------------------------------------------------------
# Retry / rotation tuning
# ---------------------------------------------------------------------------
RETRY_STATUSES = {408, 425, 500, 502, 503, 504, 529}   # transient: retry on the same key
ATTEMPTS_PER_KEY = 2                    # tries per key for transient HTTP errors
REQUEST_TIMEOUT = 60                    # seconds for one HTTP request with media attached (Gemini)
TEXT_REQUEST_TIMEOUT = 30               # seconds for one text/image request (any provider)
OPENAI_REQUEST_TIMEOUT = 40             # hard ceiling for the OpenAI-compatible providers
TOTAL_DEADLINE = 120                    # seconds for one whole ask(), all providers, models and keys

# Fairness: one slow provider must not eat the whole TOTAL_DEADLINE.
# Each provider gets max(MIN_PROVIDER_SLICE, remaining / providers_left) seconds.
MIN_PROVIDER_SLICE = 15
MAX_MODELS_PER_PROVIDER = 3             # after this many failed models, move on to the next provider

DEFAULT_429_COOLDOWN = 30               # when the provider doesn't say how long to wait
DAILY_FALLBACK_COOLDOWN = 3600          # daily quota hit but reset time unknown: re-check hourly
MAX_COOLDOWN = 24 * 3600
DEAD_KEY_COOLDOWN = 3600                # invalid / revoked / out-of-credit key: re-check hourly
FORBIDDEN_COOLDOWN = 600                # 403 that isn't clearly a dead key
MODEL_MISSING_COOLDOWN = 6 * 3600       # 404: model id retired/unknown
MODEL_OVERLOAD_COOLDOWN = 60            # model kept returning 5xx: try others first
MODEL_TIMEOUT_COOLDOWN = 60             # model timed out: it is slow right now, try others first

# True: start each request on the next key (spreads load, keeps every key under
# its per-minute limit). False: stay on one key until it hits a limit.
SPREAD_LOAD = True

BLOCK_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII",
                        "RECITATION", "IMAGE_SAFETY"}

# Phrases that mark a 429 as "daily quota used up" for the non-Gemini providers.
_DAILY_HINTS = ("per day", "daily")


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


MIN_USER_CHARS = 500  # smallest user-prompt budget worth sending to a too-small provider
MARKER = "\n[... truncated ...]\n"


def _clip_head(text: str, limit: int) -> str:
    """Shorten a system prompt by keeping its start. Its opening rules and role
    usually matter more than trailing examples, so the head is kept."""
    if limit <= 0:
        raise ValueError(f"clip limit must be positive, got {limit}")
    if len(text) <= limit:
        return text
    if limit <= len(MARKER):
        return text[:limit]
    return text[:limit - len(MARKER)] + MARKER


def _is_timeout(e: Exception) -> bool:
    """Read/connect timeouts, bare or wrapped in URLError (socket.timeout is TimeoutError on 3.10+)."""
    return isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError)


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
    text_timeout = TEXT_REQUEST_TIMEOUT
    total_deadline = TOTAL_DEADLINE
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

    def timeout_for(self, attachments: list) -> float:
        """Big media (audio/video/PDF) needs the long timeout; plain text and images don't.
        A model that hasn't answered a text prompt in 30s is not going to."""
        if any(not a.mime_type.startswith("image/") for a in attachments):
            return self.request_timeout
        return min(self.request_timeout, TEXT_REQUEST_TIMEOUT)

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
        429 / rejected keys, retries transient HTTP errors on the same key with a short
        jittered backoff. A timeout is NOT retried: the model is slow, not the key.

        `deadline` is this provider's slice, not the global deadline.

        Raises LLMUnavailable if no key could answer, LLMBlocked if the model refused the
        content."""
        label = f"{self.name}/{model}"
        body = self.build_body(model, prompt, attachments, system_prompt)
        base_timeout = self.timeout_for(attachments)

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
                    payload = self._post(model, idx, body, min(base_timeout, remaining))
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
                    if _is_timeout(e):
                        # Retrying (or rotating to another key of the same slow model) would just
                        # burn another full timeout. Give the next model / provider the time.
                        self.pool.cool_model(model, MODEL_TIMEOUT_COOLDOWN)
                        raise LLMUnavailable(f"{label}: {last_err}")
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
        sp = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        return json.dumps({
            "systemInstruction": {"parts": [{"text": sp}]},
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
    """Any OpenAI-compatible /chat/completions endpoint (Groq, Mistral, NVIDIA, OpenRouter, local)."""

    def __init__(self, name, keys, models, vision_models, spec):
        super().__init__(keys, models, vision_models)
        self.name = name
        self._url = spec["url"]
        self.max_chars = spec.get("max_chars", 100_000)
        self.max_images = spec.get("max_images", 0)
        self.quota_scope = spec.get("scope", "model")
        self.reset = spec.get("reset")
        self.extra_body = dict(spec.get("extra_body") or {})
        self.extra_headers = dict(spec.get("extra_headers") or {})

    def url(self, model, idx):
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
        sp = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": sp},
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
def _build_providers() -> list:
    built = {}

    gkeys = _as_list(_cfg("GEMINI_API_KEY"))
    if gkeys:
        built["gemini"] = _GeminiProvider(gkeys, _models_cfg("GEMINI", "MODELS", MODEL_CHAIN))

    for name, spec in _OPENAI_SPECS.items():
        if name == "local":               # <-- never in the default fallback chain _providers
            continue
        keys = _as_list(_cfg(f"{name.upper()}_API_KEY"))
        if not keys:
            continue
        models = _models_cfg(name, "MODELS", spec["models"])
        vision = _models_cfg(name, "VISION_MODELS", spec["vision"])
        built[name] = _OpenAIProvider(name, keys, models, vision, spec)

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
            "MISTRAL_API_KEY, NVIDIA_API_KEY or OPENROUTER_API_KEY in botkey.py")

    for p in providers:
        _ALL_KEYS.extend(k for k in p.keys if len(k) >= 8)
    return providers


_providers = _build_providers()
_rotate_lock = threading.Lock()
_rotate_cursor = 0
_tls = threading.local()

_local_provider = None
_local_provider_key = None


def local_provider() -> _OpenAIProvider | None:
    """The single local provider (Ollama / OpenAI-compatible endpoint) configured via
    botkey.LOCAL_MODEL, or None if unconfigured. Cached across calls.

    The local model gets its own timeouts and prompt budget from botkey (LOCAL_TIMEOUT,
    LOCAL_DEADLINE, LOCAL_MAX_CHARS). Without them it would inherit the remote defaults
    (30 s HTTP / 120 s round / 30 000 chars), which a 4B model on CPU cannot meet."""
    global _local_provider, _local_provider_key
    model = _cfg("LOCAL_MODEL") or _cfg("LOCAL_MODELS")
    if not model:
        _local_provider = None
        _local_provider_key = None
        return None
    models = tuple(_as_list(model))
    if not models:
        _local_provider = None
        _local_provider_key = None
        return None
    spec = dict(_OPENAI_SPECS.get("local", {}))
    url = _cfg("LOCAL_URL") or spec.get("url", "http://localhost:11434/v1/chat/completions")
    timeout = int(_cfg("LOCAL_TIMEOUT") or 1800)     # per HTTP call
    deadline = int(_cfg("LOCAL_DEADLINE") or 3600)   # per tool round
    budget = int(_cfg("LOCAL_MAX_CHARS") or 12_000)  # prompt chars; keep in step with num_ctx
    cache_key = (models, url, timeout, deadline, budget)
    if _local_provider is not None and _local_provider_key == cache_key:
        return _local_provider
    spec["url"] = url
    spec["models"] = list(models)
    keys = _as_list(_cfg("LOCAL_API_KEY")) or ["ollama"]
    p = _OpenAIProvider("local", keys, list(models), [], spec)
    # The local model must be able to think as long as it needs. The defaults inherited
    # from _OpenAIProvider are for cloud providers with sub-second first-token latency.
    p.request_timeout = timeout
    p.text_timeout = timeout
    p.total_deadline = deadline
    p.max_chars = budget
    _local_provider = p
    _local_provider_key = cache_key
    return _local_provider


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
    # Providers whose free tier is too small come last. Each entry carries a system-prompt
    # limit: None sends the system prompt whole; a number clips only the system prompt,
    # and only when it doesn't fit beside MIN_USER_CHARS of user prompt.
    plan = [(p, None) for p in fits]
    for p in too_big:
        sys_limit = p.max_chars - MIN_USER_CHARS
        if sys_limit <= 0:
            continue  # this provider can't hold even a minimal user prompt
        plan.append((p, sys_limit))
    if not plan:
        smallest = min(p.max_chars for p in capable)
        raise ValueError(
            f"The prompt is too large for every configured provider (smallest limit {smallest} chars, "
            f"and at least {MIN_USER_CHARS} chars must remain for the question).")

    errors, waits, perms = [], [], []
    for n, (p, sys_limit) in enumerate(plan):
        now = time.monotonic()
        if now >= deadline:
            errors.append("deadline reached")
            perms.append(False)
            break

        # Fair share of what's left, so a hanging provider can't starve the ones after it.
        # The last provider (or the only one, e.g. media -> Gemini) gets everything that remains.
        left = len(plan) - n
        slice_end = min(deadline, now + max(MIN_PROVIDER_SLICE, (deadline - now) / left))

        if sys_limit is None:
            sp, pr = system_prompt, prompt
        else:
            sp = _clip_head(system_prompt, sys_limit)
            pr = _clip(prompt, p.max_chars - len(sp))
        tried = 0
        for model in p.candidates(need_vision):
            if time.monotonic() >= slice_end or tried >= MAX_MODELS_PER_PROVIDER:
                break
            tried += 1
            try:
                text = p.call(pr, attachments, model, sp, slice_end)
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


def ask(prompt: str, attachments=None, guild_context: str = None, system_prompt: str = None) -> str:
    """
    Send a prompt, optionally with attachments (list of Attachment), and
    return the reply as a string.

    Tries the configured providers in order (see PROVIDER_ORDER), each with its
    model chain, rotating API keys on 429 / rejected keys. guild_context
    (optional) is extra text appended to the system prompt for this one call,
    used by the bot to give each Discord server its own memory.
    system_prompt (optional) overrides SYSTEM_PROMPT.

    Images work with every provider that has a vision model; audio, video and
    PDF need Gemini. Which backend answered is available from last_backend().

    Raises LLMUnavailable (retry later), LLMBlocked (content refused),
    LLMKeysRejected (all keys invalid), ValueError (unsupported attachment) or
    RuntimeError (malformed request).
    """
    attachments = list(attachments or [])
    resolved_system_prompt = build_system_prompt(guild_context, system_prompt=system_prompt)

    text_files = [a for a in attachments if a.is_text]
    attachments = [a for a in attachments if not a.is_text]
    for a in attachments:
        if not a.mime_type.startswith(SUPPORTED_PREFIXES):
            raise ValueError(f"Unsupported attachment type: {a.mime_type} ({a.name or 'unnamed'})")

    prompt = _inline_text_files(prompt or "", text_files)
    prompt = _clip(prompt, MAX_PROMPT_CHARS)

    try:
        return _generate(prompt, attachments, resolved_system_prompt)
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