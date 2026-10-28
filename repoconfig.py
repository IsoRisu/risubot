# repoconfig.py
# Config, auth, allowlists, limits, validators and the shared error types
# for the repo/PR feature. Imports nothing from the rest of the bot except botkey.
# The AI side of the feature is Gemini only; its keys/limits (GEMINI_LANES, REPO_LIMITS) are read by gemini_lanes.py.
#
# ---------------------------------------------------------------------------
# botkey.py variables (all read lazily; a missing value never breaks bot start-up,
# it only makes the repo commands fail closed with a ConfigError / NotAuthorized)
# ---------------------------------------------------------------------------
#   GITHUB_TOKEN           = "github_pat_..."   # ONE fine-grained PAT, scoped to the sandbox repo + target repos.
#                                               # Permissions: Contents RW, Pull requests RW, Metadata R.
#                                               # (Do NOT grant Workflows or Administration.)
#   REPO_OWNER_IDS         = [123456789012345678]            # Discord user ids allowed to use repo commands
#   REPO_ALLOWED_GUILD_IDS = [111111111111111111]            # whitelisted servers (falls back to ALLOWED_GUILD_ID)
#   SANDBOX_REPO           = "me/bot-sandbox"                # private scratch repo, wiped on every push
#   SANDBOX_BRANCH         = "main"                          # optional, default "main"
#   REPO_ALLOWLIST         = {                               # repos the bot may open PRs against
#       "me/app":   ["main", "dev"],                         #   -> allowed BASE branches (first = default)
#       "me/other": {"bases": ["main"], "default_base": "main"},
#   }
#   SANDBOX_OWNER_ONLY     = True    # optional: sandbox also requires REPO_OWNER_IDS (default True)
#   ALLOW_WORKFLOW_EDITS   = False   # optional: allow writing under .github/ (default False)
#   BOT_BRANCH_PREFIX      = "ai/"   # optional: every branch the bot pushes to must start with this
#
# Hard rules enforced here (the model can never override them):
#   * repo + base branch come ONLY from the user's command args, checked against REPO_ALLOWLIST;
#   * the bot only ever pushes to branches starting with BOT_BRANCH_PREFIX, never to a base branch;
#   * the bot has no merge capability (see githubclient.py).

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass

import botkey

# ---------------------------------------------------------------------------
# Limits (free tier friendly). Rationale:
#   * llmclient's OpenRouter spec caps output at 4096 tokens (~14k chars) and Groq's free
#     prompt budget is ~30k chars INCLUDING the system prompt, so single tool calls and
#     tool results are kept small; big files are built with append / edit_file.
# ---------------------------------------------------------------------------
MAX_WRITE_CALL_CHARS = 10_000      # content of ONE write_file call (longer: write first part, then append=true)
MAX_FILE_CHARS = 120_000           # a single file after appends / edits
MAX_FILES_PER_TASK = 30            # files touched (written or deleted) in one task
MAX_TOTAL_CHARS = 300_000          # sum of all staged file contents in one task
MAX_READ_RESULT_CHARS = 8_000      # one read_file result (~2k tokens); continue with start_line
MAX_LIST_ENTRIES = 150             # one list_files result
MAX_TREE_CONTEXT_CHARS = 2_500     # repo tree text put in the prompt by default
MAX_FETCH_BYTES = 1_000_000        # refuse to download repo files bigger than this at all
MAX_TOOL_ROUNDS = 12               # model <-> tool round trips per task
PROMPT_BUDGET_CHARS = 24_000       # target ceiling for the whole tool-loop prompt (Groq takes ~30k)
MAX_PATH_CHARS = 200
MAX_PATH_DEPTH = 12
MAX_PR_TITLE_CHARS = 120
MAX_PR_BODY_CHARS = 4_000
MAX_COMMIT_MSG_CHARS = 200
MAX_TASK_SECONDS = 3600            # a lock older than this is treated as stale (hung task)
HTTP_TIMEOUT = (10, 30)            # (connect, read) seconds for GitHub requests
DISCORD_MSG_LIMIT = 1900

IGNORED_DIRS = frozenset({"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build",
                          "vendor", ".idea", ".vscode", ".mypy_cache", ".pytest_cache", "target"})

DEFAULT_BRANCH_PREFIX = "ai/"


# ---------------------------------------------------------------------------
# Errors. Every error the repo feature raises on purpose derives from RepoBotError;
# repotask.format_error() turns them into the chat message + retry hint.
# ---------------------------------------------------------------------------
class RepoBotError(Exception):
    show_retry = True          # include "Retry: <command>" in the chat message
    retry_after = None         # seconds, when known

    def __init__(self, message: str, *, retry_after=None):
        super().__init__(message)
        if retry_after is not None:
            self.retry_after = retry_after

    @property
    def user_message(self) -> str:
        return str(self)


class NotAuthorized(RepoBotError):
    show_retry = False


class ConfigError(RepoBotError):
    show_retry = False


class ValidationError(RepoBotError):
    """Bad user arguments / bad model tool arguments."""


class LimitExceeded(RepoBotError):
    """A size / count limit from this module was hit."""


class SecretLeakError(RepoBotError):
    show_retry = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _cfg(name, default=None):
    return getattr(botkey, name, default)


_REPO_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
_URL_RE = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?/?$", re.I)


def normalize_repo(arg: str) -> str:
    """'owner/name', 'github.com/owner/name', 'https://github.com/owner/name.git' -> 'owner/name'."""
    s = (arg or "").strip()
    m = _URL_RE.match(s)
    if m:
        s = f"{m.group(1)}/{m.group(2)}"
    if not _REPO_RE.match(s) or s.endswith("/.") or "/.." in s:
        raise ValidationError(f"`{s[:60]}` isn't a valid repo. Use `owner/name`.")
    return s


def check_branch_name(name) -> str:
    bad = (not isinstance(name, str) or not _BRANCH_RE.match(name) or ".." in name or "//" in name
           or name.endswith(("/", ".", ".lock"))
           or any(seg.startswith(".") or seg.endswith(".lock") for seg in name.split("/")))
    if bad:
        raise ValidationError(f"Invalid branch name: `{str(name)[:60]}`.")
    return name


@dataclass(frozen=True)
class TargetPolicy:
    name: str                  # canonical spelling from REPO_ALLOWLIST
    bases: tuple               # allowed base branches
    default_base: str


@dataclass(frozen=True)
class RepoConfig:
    github_token: str
    owner_ids: frozenset
    guild_ids: frozenset
    sandbox_repo: str
    sandbox_branch: str
    targets: dict              # lower-cased "owner/name" -> TargetPolicy
    sandbox_owner_only: bool
    allow_workflow_edits: bool
    branch_prefix: str


def _id_set(value) -> frozenset:
    if value is None:
        return frozenset()
    if isinstance(value, (int, str)):
        value = [value]
    out = set()
    for v in value:
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            raise ConfigError(f"botkey: {v!r} is not a valid Discord id.")
    return frozenset(out)


def _parse_allowlist(raw) -> dict:
    if raw in (None, {}, [], ()):
        return {}
    if isinstance(raw, (list, tuple, set)):
        raw = {r: None for r in raw}
    if not isinstance(raw, dict):
        raise ConfigError("botkey.REPO_ALLOWLIST must be a dict {'owner/name': ['main', ...]}.")
    out = {}
    for key, val in raw.items():
        try:
            name = normalize_repo(str(key))
        except ValidationError as e:
            raise ConfigError(f"REPO_ALLOWLIST: {e}")
        default = None
        if val is None:
            bases = ["main"]
        elif isinstance(val, str):
            bases = [val]
        elif isinstance(val, (list, tuple, set)):
            bases = list(val)
        elif isinstance(val, dict):
            bases = list(val.get("bases") or ["main"])
            default = val.get("default_base")
        else:
            raise ConfigError(f"REPO_ALLOWLIST[{key!r}] has an unsupported value.")
        try:
            bases = [check_branch_name(b) for b in bases]
            default = check_branch_name(default or bases[0])
        except ValidationError as e:
            raise ConfigError(f"REPO_ALLOWLIST[{key!r}]: {e}")
        if default not in bases:
            raise ConfigError(f"REPO_ALLOWLIST[{key!r}]: default_base must be one of its bases.")
        out[name.lower()] = TargetPolicy(name, tuple(dict.fromkeys(bases)), default)
    return out


@functools.lru_cache(maxsize=1)
def cfg() -> RepoConfig:
    """Parsed config (cached). Raises ConfigError on malformed values."""
    sandbox = str(_cfg("SANDBOX_REPO") or "").strip()
    if sandbox:
        sandbox = normalize_repo(sandbox)
    prefix = str(_cfg("BOT_BRANCH_PREFIX") or DEFAULT_BRANCH_PREFIX)
    try:
        check_branch_name(prefix + "x")
    except ValidationError:
        raise ConfigError("botkey.BOT_BRANCH_PREFIX is not a valid branch prefix.")
    sandbox_branch = check_branch_name(str(_cfg("SANDBOX_BRANCH") or "main"))
    targets = _parse_allowlist(_cfg("REPO_ALLOWLIST"))
    if sandbox and sandbox.lower() in targets:
        raise ConfigError("SANDBOX_REPO must not also be in REPO_ALLOWLIST (it gets force-pushed).")
    guilds = _cfg("REPO_ALLOWED_GUILD_IDS")
    if guilds is None:
        guilds = _cfg("ALLOWED_GUILD_ID")          # same whitelist as !ask
    return RepoConfig(
        github_token=str(_cfg("GITHUB_TOKEN") or "").strip(),
        owner_ids=_id_set(_cfg("REPO_OWNER_IDS")),
        guild_ids=_id_set(guilds),
        sandbox_repo=sandbox,
        sandbox_branch=sandbox_branch,
        targets=targets,
        sandbox_owner_only=bool(_cfg("SANDBOX_OWNER_ONLY", True)),
        allow_workflow_edits=bool(_cfg("ALLOW_WORKFLOW_EDITS", False)),
        branch_prefix=prefix,
    )


def require_github_token() -> str:
    token = cfg().github_token
    if not token:
        raise ConfigError("GITHUB_TOKEN is not set in botkey.py.")
    return token


def require_sandbox_repo() -> str:
    repo = cfg().sandbox_repo
    if not repo:
        raise ConfigError("SANDBOX_REPO is not set in botkey.py.")
    return repo


# ---------------------------------------------------------------------------
# Authorization (fails closed: empty lists authorize nobody)
# ---------------------------------------------------------------------------
def authorize(user_id: int, guild_id, *, mode: str = "target") -> None:
    """Raise NotAuthorized unless this user may run a repo task in this server.
    mode: "target" (always owner-only) or "sandbox" (owner-only unless SANDBOX_OWNER_ONLY=False)."""
    c = cfg()
    if guild_id is None or guild_id not in c.guild_ids:
        raise NotAuthorized("This command isn't available here.")
    if (mode != "sandbox" or c.sandbox_owner_only) and user_id not in c.owner_ids:
        raise NotAuthorized("Only the bot owner can use repo commands.")


# ---------------------------------------------------------------------------
# Repo / branch validation (the ONLY source of repo + branch is the user's command)
# ---------------------------------------------------------------------------
def policy_for(repo: str) -> TargetPolicy:
    name = normalize_repo(repo)
    policy = cfg().targets.get(name.lower())
    if policy is None:
        raise ValidationError(f"`{name}` is not on the bot's allowed repo list.")
    return policy


@dataclass(frozen=True)
class TargetSpec:
    repo: str      # canonical spelling from the allowlist
    base: str      # validated base branch


def resolve_target(repo_arg: str, base_arg: str | None = None) -> TargetSpec:
    policy = policy_for(repo_arg)
    base = check_branch_name(base_arg) if base_arg else policy.default_base
    if base not in policy.bases:
        raise ValidationError(f"Base branch `{base}` isn't allowed for `{policy.name}` "
                              f"(allowed: {', '.join(policy.bases)}).")
    return TargetSpec(policy.name, base)


def validate_head_branch(name: str, policy: TargetPolicy) -> str:
    """Branch the bot may push to: must carry the bot prefix and never be a base branch."""
    prefix = cfg().branch_prefix
    check_branch_name(name)
    if not name.startswith(prefix) or len(name) <= len(prefix):
        raise ValidationError(f"Bot branches must start with `{prefix}` (got `{name[:60]}`).")
    if name in policy.bases:
        raise ValidationError("The bot never pushes to a base branch.")
    return name


def assert_pushable(repo: str, branch: str, base: str) -> TargetPolicy:
    """Single choke point used by githubclient before any write to a target repo."""
    policy = policy_for(repo)
    if base not in policy.bases:
        raise ValidationError(f"Base branch `{base}` isn't allowed for `{policy.name}`.")
    validate_head_branch(branch, policy)
    return policy


# ---------------------------------------------------------------------------
# Text sanitizing for things that end up on GitHub
# ---------------------------------------------------------------------------
def sanitize_pr_text(title: str | None, body: str | None, *, default_title: str = "AI-assisted changes") -> tuple:
    """One-line title <=120 chars, body <=4000 chars, @mentions defused, secrets refused."""
    t = " ".join((title or "").split())[:MAX_PR_TITLE_CHARS] or default_title
    b = (body or "").strip()[:MAX_PR_BODY_CHARS]
    for text in (t, b):
        label = find_secret(text)
        if label:
            raise SecretLeakError(f"Refusing to put {label} into the pull request text.")
    defuse = lambda s: re.sub(r"(?<![\w`])@(?=[A-Za-z0-9])", "@\u200b", s)
    return defuse(t), defuse(b)


def sanitize_commit_message(msg: str | None, default: str = "AI-assisted changes") -> str:
    first, _, rest = (msg or "").strip().partition("\n")
    first = " ".join(first.split())[:MAX_COMMIT_MSG_CHARS] or default
    text = first + (("\n\n" + rest.strip()[:MAX_PR_BODY_CHARS]) if rest.strip() else "")
    if find_secret(text):
        raise SecretLeakError("Refusing to put a secret into a commit message.")
    return text


# ---------------------------------------------------------------------------
# Secrets: never written to GitHub, never echoed to chat
# ---------------------------------------------------------------------------
_SECRET_NAME_RE = re.compile(r"key|token|secret|passw|lanes|^pat$|_pat$", re.I)   # GEMINI_LANES holds API keys
_SECRET_PATTERNS = (
    ("a GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}")),
    ("a Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("a Groq API key", re.compile(r"gsk_[A-Za-z0-9]{40,}")),
    ("an NVIDIA API key", re.compile(r"nvapi-[A-Za-z0-9_\-]{40,}")),
    ("an OpenRouter API key", re.compile(r"sk-or-[A-Za-z0-9_\-]{20,}")),
    ("a Discord bot token", re.compile(r"[MNO][A-Za-z0-9_\-]{23,25}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{27,38}")),
    ("an AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)


def _flatten_strings(v):
    """Strings inside a str / list / tuple / set / dict of those, at any depth (GEMINI_LANES is a list of lists)."""
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _flatten_strings(x)
    elif isinstance(v, (list, tuple, set, frozenset)):
        for x in v:
            yield from _flatten_strings(x)


@functools.lru_cache(maxsize=1)
def secret_values() -> tuple:
    vals = set()
    for name in dir(botkey):
        if name.startswith("_") or not _SECRET_NAME_RE.search(name):
            continue
        vals.update(s for s in _flatten_strings(getattr(botkey, name, None)) if len(s) >= 8)
    return tuple(vals)


def find_secret(text: str | None) -> str | None:
    """Label of the first secret found in text, or None. Never returns the secret itself."""
    if not text:
        return None
    for v in secret_values():
        if v in text:
            return "a bot secret from botkey.py"
    for label, rx in _SECRET_PATTERNS:
        if rx.search(text):
            return label
    return None


def redact(text: str) -> str:
    for v in secret_values():
        text = text.replace(v, "***")
    for _, rx in _SECRET_PATTERNS:
        text = rx.sub("***", text)
    return text


# ---------------------------------------------------------------------------
# Repo path validation (model-supplied paths are untrusted)
# ---------------------------------------------------------------------------
_SECRET_FILENAMES = frozenset({"botkey.py", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_dsa", "id_ecdsa",
                               "id_ed25519", "credentials", "secrets.json", "secrets.yml", "secrets.yaml"})
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")
_ENV_SAFE_SUFFIXES = (".example", ".sample", ".template", ".dist")
BINARY_EXTENSIONS = frozenset({
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "tiff", "pdf", "zip", "gz", "tar", "tgz", "7z", "rar",
    "exe", "dll", "so", "dylib", "bin", "class", "jar", "pyc", "o", "a", "woff", "woff2", "ttf", "otf", "eot",
    "mp3", "mp4", "wav", "ogg", "webm", "mov", "avi", "sqlite", "db", "psd",
})


def normalize_path(path, *, for_write: bool = False) -> str:
    """Validate a repo-relative path from the model / user. Returns the clean path or raises ValidationError."""
    if not isinstance(path, str):
        raise ValidationError("path must be a string.")
    p = unicodedata.normalize("NFC", path.strip().replace("\\", "/"))
    while p.startswith("./"):
        p = p[2:]
    if not p or p.startswith("/") or len(p) > MAX_PATH_CHARS:
        raise ValidationError(f"Invalid path `{p[:60]}` (must be relative, 1-{MAX_PATH_CHARS} chars).")
    if any(ord(c) < 32 or ord(c) == 127 for c in p):
        raise ValidationError("Path contains control characters.")
    segs = p.split("/")
    if any(s in ("", ".", "..") or s != s.strip() for s in segs) or len(segs) > MAX_PATH_DEPTH:
        raise ValidationError(f"Invalid path `{p[:60]}`.")
    if any(s.lower() == ".git" for s in segs):
        raise ValidationError("Paths inside .git are not allowed.")
    name = segs[-1].lower()
    is_env = name == ".env" or (name.startswith(".env.") and not name.endswith(_ENV_SAFE_SUFFIXES))
    if name in _SECRET_FILENAMES or name.endswith(_SECRET_SUFFIXES) or is_env:
        raise ValidationError(f"`{segs[-1]}` looks like a secrets file; the bot won't touch it.")
    if for_write:
        if segs[0].lower() == ".github" and not cfg().allow_workflow_edits:
            raise ValidationError("Writing under .github/ is disabled (ALLOW_WORKFLOW_EDITS=False).")
        if name == ".gitmodules":
            raise ValidationError(".gitmodules can't be modified by the bot.")
        ext = name.rsplit(".", 1)[-1] if "." in name else ""
        if ext in BINARY_EXTENSIONS:
            raise ValidationError(f"Binary files (.{ext}) can't be written; text files only.")
    return p


def is_ignored_path(path: str) -> bool:
    return any(seg in IGNORED_DIRS for seg in path.split("/")[:-1])
