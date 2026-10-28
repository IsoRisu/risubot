# repocmds.py
# Discord command for the sandbox flow.  `!code [--branch <name>] <what to build>`
#
# Wiring in risubot.py (two lines, e.g. just above "# --- Run the Bot ---"):
#     import repocmds
#     repocmds.register(bot)
#
# Flow: authorize -> take the task lock -> Gemini pipeline (repo_pipeline.run_task) -> if files were produced:
# force-push ONE orphan commit to the sandbox repo -> post the commit URL.
# If the pipeline produced no files, nothing is pushed and its summary is posted (the sandbox is not wiped).
#
# The AI side is Gemini only (account lanes + quota live in gemini_lanes / gemini_client):
#   plan call -> bot reads the requested files -> edit call(s) -> deterministic applier -> <= 1 repair call.
# The model never touches git: this module takes TaskResult.changes and pushes them itself
# (sandbox_force_push here, commit_to_branch / open_pr in repotarget). The sandbox repo is empty from the
# model's point of view (nothing to read), so everything it produces is a "create".
#
# This module also holds the glue the /repo command (repotarget.py) shares:
#   Backend / default_backend()   the Gemini quota store + lanes (built lazily from botkey, one per process)
#   run_pipeline(...)             repo_pipeline.run_task with that backend
#   make_reader(view)             read_file(path) -> str | None for the pipeline, on top of GitHubClient.read_file
#   build_notes(read_file)        README.md + AGENTS.md, truncated to MAX_NOTES_CHARS together
# Test hook: pass backend=Backend(store, lanes, call) to replace the Gemini call (call=None uses the real client;
# pass functools.partial(gemini_client.call_gemini, transport=...) to script the HTTP layer instead).

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from dataclasses import dataclass

import githubclient as gh
import repo_pipeline as rp
import repoconfig as rc
import repotask

log = logging.getLogger("repocmds")

MAX_PROMPT_CHARS = 6_000            # user prompt + inlined text attachments
MAX_ATTACH_FILES = 3
MAX_ATTACH_CHARS = 20_000           # per attached text file
MAX_REPLY_CHARS = 900               # the AI's summary shown next to the URL
MAX_NOTES_CHARS = 6_000             # README.md + AGENTS.md together, as sent to the planning call
NOTE_FILES = ("README.md", "AGENTS.md")
NOTES_CUT = "\n[... truncated]"
QUOTA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_quota.json")

SANDBOX_TREE = (
    "(empty: this is a fresh sandbox repository without any files. Create every file you need as a new file; "
    "need_files must be [].)")

CODE_USAGE = ("Usage: `!code [--branch <name>] <what to build>` — the AI writes files and I push them to the sandbox "
              "repo (it is wiped each time). Text/code files can be attached as input.")

_BRANCH_FLAG = re.compile(r"^\s*--branch(?:=|\s+)(\S+)\s*", re.I)


# ---------------------------------------------------------------------------
# Gemini backend (shared with repotarget)
# ---------------------------------------------------------------------------
@dataclass
class Backend:
    """What repo_pipeline.run_task needs besides the request: the quota store, the lanes, and the call hook."""
    store: object
    lanes: list
    call: object = None             # None: gemini_client.call_gemini (looked up by the pipeline at call time)


_backend: Backend | None = None
_backend_lock = threading.Lock()


def default_backend() -> Backend:
    """The process-wide backend: lanes from botkey.GEMINI_LANES / GEMINI_API_KEY, limits from REPO_LIMITS,
    daily counters persisted in QUOTA_FILE. Built on first use so a missing key is reported in chat, not at startup.
    Raises rc.ConfigError (no retry hint) when the configuration is missing or invalid."""
    global _backend
    with _backend_lock:
        if _backend is None:
            import gemini_lanes as gl
            try:
                lanes = gl.load_lanes()
                rpd, rpm = gl.load_limits()
                store = gl.QuotaStore(lanes, QUOTA_FILE, rpd=rpd, rpm=rpm) if lanes else None
            except ValueError as e:
                raise rc.ConfigError(f"The Gemini configuration in botkey.py is invalid: {e}") from e
            if store is None:
                raise rc.ConfigError("No Gemini API key is configured. The bot owner must set GEMINI_LANES "
                                     "(or GEMINI_API_KEY) in botkey.py.")
            _backend = Backend(store, lanes)
        return _backend


def run_pipeline(request: str, tree_text: str, notes_text: str, read_file, backend: Backend | None = None
                 ) -> rp.TaskResult:
    """Blocking. repo_pipeline.run_task on `backend` (default: default_backend()). Raises the pipeline's typed
    errors (all rc.RepoBotError); repotask.run_task turns them into the chat message."""
    backend = backend or default_backend()
    extra = {"call": backend.call} if backend.call is not None else {}
    return rp.run_task(request, tree_text, notes_text, read_file, backend.store, backend.lanes, **extra)


# ---------------------------------------------------------------------------
# Reading the repository for the pipeline
# ---------------------------------------------------------------------------
def make_reader(view):
    """read_file(path) -> str | None for repo_pipeline, on top of a githubclient.RepoView (which reads through
    GitHubClient.read_file). None = the file does not exist. Binary files raise rc.ValidationError; files that are
    too large raise rc.LimitExceeded (both are handled by the pipeline: the file is listed as "not shown").
    Any other GitHub error (auth, rate limit, outage) propagates and fails the task."""
    def read_file(path: str):
        try:
            fc = view.read(path)
        except gh.GitHubNotFound:
            return None
        if fc.is_binary:
            raise rc.ValidationError(f"`{path}` is a binary file.")
        return fc.text
    return read_file


def build_notes(read_file, limit: int = MAX_NOTES_CHARS) -> str:
    """README.md and AGENTS.md (the ones that exist) as labelled blocks, at most `limit` chars in total, headers
    included. The budget is shared fairly: a short file keeps all of its text, the rest is split evenly, and a cut
    file ends with a visible marker. Unreadable files are skipped. Returns "" when there is nothing."""
    docs = []
    for name in NOTE_FILES:
        try:
            text = read_file(name)
        except rc.RepoBotError as e:
            log.info("notes file %s skipped: %s", name, e)
            continue
        if text and text.strip():
            docs.append((name, text.strip()))
    if not docs:
        return ""
    heads = [f"=== {name} ===\n" for name, _ in docs]
    room = max(0, limit - sum(len(h) for h in heads) - 2 * (len(docs) - 1))
    caps, left = {}, room
    for k, i in enumerate(sorted(range(len(docs)), key=lambda j: len(docs[j][1]))):
        caps[i] = min(len(docs[i][1]), left // (len(docs) - k))
        left -= caps[i]
    parts = []
    for i, (_, text) in enumerate(docs):
        if caps[i] >= len(text):
            body = text
        elif caps[i] > len(NOTES_CUT):
            body = text[:caps[i] - len(NOTES_CUT)] + NOTES_CUT
        else:
            body = text[:caps[i]]
        parts.append(heads[i] + body)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
def parse_code_args(raw: str) -> tuple:
    """'--branch x do things' -> ('x', 'do things'). Branch is validated; None means the configured default."""
    branch = None
    m = _BRANCH_FLAG.match(raw or "")
    if m:
        branch = rc.check_branch_name(m.group(1))
        raw = raw[m.end():]
    return branch, (raw or "").strip()


def format_success(commit: gh.CommitResult, files: list, reply: str) -> str:
    shown = ", ".join(f"`{f}`" for f in files[:8]) + (f" (+{len(files) - 8} more)" if len(files) > 8 else "")
    text = f"✅ Pushed {len(files)} file(s) to the sandbox: {shown}\n{commit.url}"
    if reply:
        text += "\n" + reply.strip()[:MAX_REPLY_CHARS]
    return text


async def sandbox_job(prompt: str, branch: str | None, *, client: gh.GitHubClient | None = None,
                      backend: Backend | None = None) -> str:
    """The work done while holding the lock. Returns the message to post. Raises on any failure.

    `backend` is the test hook (see the module header). The sandbox has nothing to read, so the pipeline gets an
    empty tree, no notes and a reader that finds no file; whatever it returns is pushed as one orphan commit."""
    client = client or gh.GitHubClient()

    def blocking() -> str:
        result = run_pipeline(prompt, SANDBOX_TREE, "", lambda path: None, backend)
        files = {p: t for p, t in result.changes.items() if t is not None}     # sandbox pushes are files only
        if not files:
            return result.summary or "(The AI produced no files and no reply.)"
        commit = client.sandbox_force_push(files, f"Sandbox output ({len(files)} file(s))", branch)
        return format_success(commit, sorted(files), result.summary)

    return await asyncio.to_thread(blocking)


async def _attached_text(message) -> str:
    """Text/code attachments of the command message, appended to the prompt as labelled data."""
    import llmclient
    blocks = []
    for a in message.attachments[:MAX_ATTACH_FILES]:
        if not llmclient.is_text_filename(a.filename) or a.size > llmclient.MAX_TEXT_FILE_BYTES:
            continue
        try:
            text = llmclient.Attachment.from_bytes(await a.read(), a.filename).decode_text()
        except ValueError:
            continue
        blocks.append(f"=== ATTACHED FILE (data): {a.filename} ===\n{text[:MAX_ATTACH_CHARS]}\n=== END ===")
    return ("\n\n" + "\n\n".join(blocks)) if blocks else ""


def register(bot, *, client_factory=gh.GitHubClient, backend: Backend | None = None):
    """Adds the `!code` (alias `!scratch`) command. Gemini only: plan -> read -> edit -> apply -> at most one repair,
    then ONE orphan commit force-pushed to the sandbox repo (never any other repo). `backend` is the test hook."""
    from discord.ext import commands

    @commands.guild_only()
    @bot.command(name="code", aliases=["scratch"],
                 help="AI writes files and pushes them to the sandbox repo. Usage: !code [--branch <name>] <what to build>")
    async def code_command(ctx, *, raw: str = ""):
        retry = (ctx.message.content or "")[:1000]
        if ctx.message.attachments:
            retry += "  (re-attach your files)"
        try:
            rc.authorize(ctx.author.id, ctx.guild.id, mode="sandbox")
            branch, prompt = parse_code_args(raw)
            prompt += await _attached_text(ctx.message)
            if not prompt.strip():
                await ctx.reply(CODE_USAGE)
                return
            if len(prompt) > MAX_PROMPT_CHARS:
                raise rc.ValidationError(f"The request is too long ({len(prompt)} chars; max {MAX_PROMPT_CHARS}).")
        except rc.RepoBotError as e:
            await ctx.reply(repotask.format_error(e, retry))
            return

        task = repotask.TaskContext("sandbox", ctx.author.id, ctx.guild.id, ctx.channel.id, retry, prompt[:80])

        async def work(_task):
            return await sandbox_job(prompt, branch, client=client_factory(), backend=backend)

        async with ctx.typing():
            await repotask.run_task(task, ctx.reply, work)

    return code_command
