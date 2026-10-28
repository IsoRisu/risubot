# repotarget.py
# TARGET repositories.
#   /repo task repo:<owner/name> prompt:<text> [base:<branch>] [branch:<name>] [iter:<bool>] [title:<text>] [body:<text>]
#
# Owner-only slash command. Gemini (only) plans and edits an allowlisted GitHub repo through the
# repo_pipeline: the bot sends the file tree + README/AGENTS.md to a planning call, reads the files it asks for,
# sends them to an edit call, applies the edits deterministically (at most one repair call), and verifies the
# result. The edits are committed to a BOT BRANCH ("ai/<name>", never the base branch) and a pull request against
# the base is opened. Nothing changed means nothing is pushed. The model never touches git: this module takes
# TaskResult.changes and calls commit_to_branch / open_pr itself. `iter` continues on an existing bot branch
# (the named one, else the newest). The library functions default to read_only=True (the proposed edits are
# discarded with a notice); the slash command passes read_only=False.
#
# Wiring in risubot.py (two lines, e.g. just above "# --- Run the Bot ---"):
#     import repotarget
#     repotarget.register(bot)
#
# Slash commands only appear after the command tree is synced. register() chains a sync into
# bot.setup_hook (to the guilds in botkey.REPO_ALLOWED_GUILD_IDS, which is instant; globally if that
# list is empty). The bot must have been invited with the `applications.commands` scope.
#
# Flow: authorize + resolve repo/branch (local, fast; failures are ephemeral) -> defer the interaction
# (the pipeline takes far longer than Discord's 3 s) -> task lock -> [thread] view, tree, notes, pipeline,
# commit + PR -> post the answer with interaction.followup.send (chunked; falls back to a plain channel message
# if the 15-minute interaction token has expired). Errors (RepoBotError and the pipeline's typed errors, e.g.
# quota exhausted with the earliest retry time across the Gemini lanes) are rendered by repotask.format_error.
#
# Shared glue (Gemini backend, file reader, notes) lives in repocmds.py. Test hook: backend=repocmds.Backend(...).

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import discord
from discord import app_commands

import githubclient as gh
import repo_pipeline as rp
import repocmds
import repoconfig as rc
import repotask

try:
    import botkey
except ImportError:                      # tests / tools that import this module without a botkey
    botkey = None

log = logging.getLogger("repotarget")

MAX_PROMPT_CHARS = 6_000            # the /repo task prompt (same limit as !code)
MAX_REPLY_CHARS = 6_000             # the AI's summary as shown in Discord (about 3-4 messages)
DISCORD_CHUNK = 1_900               # Discord's cap is 2000; leave headroom
RETRY_COMMAND_CHARS = 1_000
NO_MENTIONS = discord.AllowedMentions.none()      # model output must never ping anyone
FILE_MODES = frozenset({"100644", "100755"})      # modes that are carried over to the commit

READ_ONLY_NOTE = ("This run is READ-ONLY: nothing you change will be saved. Put your answer to the request in "
                  "\"summary\" and send \"edits\": [].")
WRITE_NOTE = ("You may change files. The bot commits your edits to a new branch and opens the pull request itself. "
              "If no change is needed, send \"edits\": [] and explain in \"summary\".")

MAX_TITLE_CHARS = 200               # /repo task title option
MAX_BODY_CHARS = 1_500              # /repo task body option
DEFAULT_TITLE_CHARS = 72


# ---------------------------------------------------------------------------
# The request text sent to the pipeline
# ---------------------------------------------------------------------------
def build_request(spec, prompt: str, *, read_only: bool = True, note: str = "") -> str:
    """The user's request with a one-paragraph header: repository, base branch, optional `note` (e.g. which bot
    branch is being continued) and whether edits will be kept."""
    header = f"Repository: {spec.repo} (base branch: {spec.base})."
    if note:
        header += " " + note
    header += " " + (READ_ONLY_NOTE if read_only else WRITE_NOTE)
    return header + "\n\nRequest:\n" + prompt


# ---------------------------------------------------------------------------
# Branches, commit and PR
#
# GitHubClient methods used here (see githubclient.py):
#   view(repo, ref) -> RepoView          read-only handle (validates `ref`); .listing -> list_tree, .read -> read_file
#   latest_bot_branch(repo) -> str | None                       branch of the newest open bot PR
#   commit_to_branch(repo, branch, base, changes, message, modes=None) -> CommitResult
#         creates `branch` from `base` when it doesn't exist yet, else adds a commit on top of it.
#   open_pr(repo, head, base, title, body) -> PullRequest       idempotent: reuses an open PR for `head`
# ---------------------------------------------------------------------------
@dataclass
class Published:
    """What was pushed for one run."""
    branch: str
    files: list
    commit_url: str = ""
    pr_url: str = ""
    pr_error: str = ""


def bot_branch_name(raw: str, base: Optional[str] = None) -> str:
    """User-supplied branch name -> 'ai/<name>'. The prefix is forced, so the bot can never write to
    main or any other allowlisted base branch. Raises rc.ValidationError for bad names."""
    prefix = rc.cfg().branch_prefix
    name = (raw or "").strip()
    if name.startswith(prefix):
        name = name[len(prefix):]
    name = rc.check_branch_name(name)
    full = prefix + name
    if full == base:
        raise rc.ValidationError("That branch name is the base branch.")
    return full


def new_branch_name(prompt: str, now: Optional[float] = None) -> str:
    """'ai/<first words of the prompt>-<MMDD-HHMMSS>'."""
    words = re.findall(r"[a-z0-9]+", (prompt or "").lower())[:4]
    slug = "-".join(words)[:30].strip("-") or "task"
    return f"{rc.cfg().branch_prefix}{slug}-{time.strftime('%m%d-%H%M%S', time.localtime(now))}"


def _defang(text: str) -> str:
    """Stop @mentions in prompt/model text from notifying people on GitHub."""
    return (text or "").replace("@", "@\u200b")


def default_title(prompt: str) -> str:
    for line in (prompt or "").splitlines():
        line = " ".join(line.split())
        if line:
            return line if len(line) <= DEFAULT_TITLE_CHARS else line[:DEFAULT_TITLE_CHARS - 1].rstrip() + "…"
    return "AI changes"


def default_body(prompt: str, reply: str) -> str:
    parts = ["Opened by the Discord bot from a `/repo task` request.",
             "**Request**\n\n" + _defang((prompt or "").strip())[:1_500]]
    if (reply or "").strip():
        parts.append("**AI summary**\n\n" + _defang(reply.strip())[:1_500])
    return "\n\n".join(parts)


def publish_changes(client, run: "TargetRun", prompt: str, title: Optional[str] = None,
                    body: Optional[str] = None) -> Published:
    """Commit run.result.changes to run.branch and open the PR. A commit failure raises (nothing was pushed,
    run_task reports it). A PR failure after a successful commit is recorded in pr_error instead, so the branch
    name is not lost. Secret scanning and path validation happen inside commit_to_branch."""
    spec = run.spec
    files = run.result.changes
    title = (title or "").strip() or default_title(prompt)
    commit = client.commit_to_branch(spec.repo, branch=run.branch, base=spec.base, changes=files, modes=run.modes,
                                     message=f"{title}\n\nRequested via Discord /repo task.")
    pub = Published(run.branch, sorted(files), commit_url=str(getattr(commit, "url", "") or ""))
    body_text = _defang(body.strip()) if (body or "").strip() else default_body(prompt, run.result.summary)
    try:
        pr = client.open_pr(spec.repo, head=run.branch, base=spec.base, title=title, body=body_text)
        pub.pr_url = str(getattr(pr, "url", "") or pr)
    except Exception as e:                       # the commit is already on GitHub: report, do not lose it
        log.exception("open_pr failed after a successful commit to %s", run.branch)
        pub.pr_error = f"{type(e).__name__}: {e}"[:300]
    return pub


@dataclass
class TargetRun:
    """Everything the blocking run produced."""
    spec: object
    result: rp.TaskResult
    read_only: bool = True
    branch: Optional[str] = None            # bot branch written to (write mode) or continued (iter)
    iterated: bool = False
    modes: Optional[dict] = None            # {path: git mode} of changed files that already exist (keeps exec bits)
    published: Optional[Published] = None   # None: nothing was committed


def existing_modes(listing, changes: dict) -> dict:
    """Git modes of the changed files that exist in `listing`, so an overwrite keeps e.g. the executable bit."""
    wanted = {p for p, text in changes.items() if text is not None}
    return {e.path: e.mode for e in listing.entries if e.path in wanted and e.mode in FILE_MODES}


def run_target(spec, prompt: str, *, client, backend: Optional[repocmds.Backend] = None, read_only: bool = True,
               branch: Optional[str] = None, iterate: bool = False, title: Optional[str] = None,
               body: Optional[str] = None) -> TargetRun:
    """Blocking (run it with asyncio.to_thread). Raises on any failure.

    iterate=True: the view comes from `branch` (already 'ai/'-prefixed) or else the newest bot branch, and
    new edits are committed on top of it. Otherwise the view is the base branch and write mode commits to
    `branch` or a generated 'ai/...' name. read_only=True never commits.

    `backend` is the test hook (repocmds.Backend: quota store, lanes, call). The Gemini calls are made by
    repo_pipeline.run_task; this function only reads the repo (tree, notes, requested files) and, in write mode,
    commits what the pipeline returns."""
    ref, note = spec.base, ""
    if iterate:
        branch = branch or client.latest_bot_branch(spec.repo)
        if not branch:
            raise rc.ValidationError("There is no bot branch to continue on yet. Run the task without `iter` first.")
        ref = branch
        note = f"Continuing on the existing branch `{branch}`; the repository files you see are its current state."
    elif not read_only:
        branch = branch or new_branch_name(prompt)
    view = client.view(spec.repo, ref)
    listing = view.listing
    tree_text = listing.render()
    read_file = repocmds.make_reader(view)
    notes_text = repocmds.build_notes(read_file)
    result = repocmds.run_pipeline(build_request(spec, prompt, read_only=read_only, note=note), tree_text,
                                   notes_text, read_file, backend)
    run = TargetRun(spec, result, read_only=read_only, branch=branch, iterated=iterate,
                    modes=existing_modes(listing, result.changes))
    if not read_only and result.changes:
        run.published = publish_changes(client, run, prompt, title, body)
    return run


def publish_status(run: TargetRun) -> str:
    """The outcome line of a write run: PR link, partial failure, or 'no changes'."""
    pub = run.published
    if pub is None:
        return "ℹ️ No changes: the AI proposed no file edits, so nothing was committed and no PR was opened."
    shown = ", ".join(f"`{f}`" for f in pub.files[:8]) + (f" (+{len(pub.files) - 8} more)" if len(pub.files) > 8 else "")
    if pub.pr_url:
        return f"✅ PR: {pub.pr_url}\nBranch `{pub.branch}` · {len(pub.files)} file(s): {shown}"
    commit = f" ({pub.commit_url})" if pub.commit_url else ""
    return (f"⚠️ Committed {len(pub.files)} file(s) to branch `{pub.branch}`{commit}, but opening the PR failed: "
            f"{pub.pr_error or 'unknown error'}\nOpen the PR by hand, or run the task again with `iter` to retry.")


def format_result(run: TargetRun) -> str:
    """The message to post for a finished run."""
    res = run.result
    calls = f"{len(res.calls)} AI call(s)"
    if run.read_only:
        head = f"📖 **{run.spec.repo}** @ `{run.spec.base}` · read-only · {calls}"
    else:
        head = (f"🛠️ **{run.spec.repo}** @ `{run.spec.base}` · branch `{run.branch}`"
                f"{' (continued)' if run.iterated else ''} · {calls}")
    body = (res.summary or "").strip() or "(The AI gave no written reply.)"
    if len(body) > MAX_REPLY_CHARS:
        body = body[:MAX_REPLY_CHARS].rstrip() + "\n… (reply truncated)"
    if not run.read_only:
        return head + "\n" + publish_status(run) + "\n\n" + body
    text = head + "\n" + body
    if res.changes:
        names = sorted(res.changes)
        shown = ", ".join(f"`{n}`" for n in names[:5]) + (f" (+{len(names) - 5} more)" if len(names) > 5 else "")
        text += (f"\n\n⚠️ The AI prepared changes to {len(names)} file(s) ({shown}) but this command is read-only "
                 "here, so nothing was committed.")
    return text


async def target_job(spec, prompt: str, *, client=None, backend: Optional[repocmds.Backend] = None,
                     read_only: bool = True, branch: Optional[str] = None, iterate: bool = False,
                     title: Optional[str] = None, body: Optional[str] = None) -> str:
    """The work done while holding the task lock. Returns the message to post. Raises on any failure."""
    client = client or gh.GitHubClient()
    run = await asyncio.to_thread(run_target, spec, prompt, client=client, backend=backend, read_only=read_only,
                                  branch=branch, iterate=iterate, title=title, body=body)
    return format_result(run)


# ---------------------------------------------------------------------------
# Discord plumbing
# ---------------------------------------------------------------------------
def split_message(text: str, limit: int = DISCORD_CHUNK) -> list:
    """Split text for Discord's 2000-char cap, preferring line breaks. A cut inside a ``` fence closes
    it and reopens it in the next chunk. (Same algorithm as risubot.chunk_text, which cannot be imported
    from here without a circular import.)"""
    chunks, open_fence = [], None
    while text:
        prefix = open_fence + "\n" if open_fence else ""
        room = limit - len(prefix) - 4
        if len(text) <= room:
            chunks.append(prefix + text)
            break
        cut = text.rfind("\n", 0, room)
        if cut <= 0:
            cut = room
        body, text = prefix + text[:cut], text[cut:].lstrip("\n")
        state = None
        for line in body.splitlines():
            if line.startswith("```"):
                state = None if state else line.strip()
        if state:
            body += "\n```"
        open_fence = state
        chunks.append(body)
    return [c for c in chunks if c.strip()]


def retry_text(repo: str, prompt: str, base: Optional[str], branch: Optional[str] = None,
               iterate: bool = False, title: Optional[str] = None) -> str:
    """What the user would type again; shown by repotask.format_error as "Retry with: ..."."""
    text = f"/repo task repo:{repo} prompt:{prompt}" + (f" base:{base}" if base else "")
    if branch:
        text += f" branch:{branch}"
    if iterate:
        text += " iter:True"
    if title:
        text += f" title:{title}"
    return text[:RETRY_COMMAND_CHARS]


def make_sender(interaction):
    """send(text) for run_task: posts with followup.send, split to Discord's limit. If the interaction
    token is gone (15 minutes after the command) the rest goes to the channel as ordinary messages."""
    state = {"expired": False}

    async def send(text: str):
        for chunk in split_message(str(text)):
            if not state["expired"]:
                try:
                    await interaction.followup.send(chunk, allowed_mentions=NO_MENTIONS)
                    continue
                except discord.HTTPException as e:
                    if e.status not in (401, 404):               # token expired / webhook unknown
                        raise
                    state["expired"] = True
                    log.warning("interaction token expired; posting to the channel instead")
                    chunk = f"{interaction.user.mention} (the slash command timed out, posting here)\n{chunk}"[:2000]
            channel = interaction.channel or interaction.client.get_partial_messageable(interaction.channel_id)
            await channel.send(chunk, allowed_mentions=NO_MENTIONS)
    return send


async def _respond_ephemeral(interaction, text: str):
    text = text[:DISCORD_CHUNK]
    if interaction.response.is_done():
        await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
    else:
        await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)


def configured_guild_ids() -> list:
    """botkey.REPO_ALLOWED_GUILD_IDS as ints; [] when unset (then the command is registered globally)."""
    try:
        return [int(g) for g in (getattr(botkey, "REPO_ALLOWED_GUILD_IDS", None) or [])]
    except (TypeError, ValueError):
        return []


async def sync_commands(bot, guild_ids: list) -> None:
    """Push the slash commands to Discord. Never raises: a failed sync must not stop the bot."""
    targets = [discord.Object(id=g) for g in guild_ids] or [None]
    for guild in targets:
        try:
            synced = await (bot.tree.sync(guild=guild) if guild else bot.tree.sync())
            log.info("synced %d slash command(s) to %s", len(synced), f"guild {guild.id}" if guild else "all guilds")
        except Exception as e:
            log.warning("slash command sync to %s failed: %s: %s (is the bot in that guild, and invited with "
                        "the applications.commands scope?)", f"guild {guild.id}" if guild else "global",
                        type(e).__name__, e)


def register(bot, *, client_factory=gh.GitHubClient, backend: Optional[repocmds.Backend] = None, sync: bool = True):
    """Adds the /repo command group to bot.tree and (sync=True) syncs it on startup. Returns the group.

    /repo task is Gemini only: plan -> read the requested files -> edit -> apply -> at most one repair -> verify,
    then the bot commits to an `ai/` branch and opens the PR (the model never touches git). `backend` is the test
    hook (quota store, lanes, Gemini call); by default it is built from botkey on first use."""
    guild_ids = configured_guild_ids()
    group = app_commands.Group(name="repo", description="AI work on allowlisted GitHub repos (owner only)",
                               guild_only=True)

    @group.command(name="task", description="Ask Gemini to change a repo; edits become a branch + pull request")
    @app_commands.describe(repo="owner/name of an allowlisted repository",
                           prompt="What should the AI look into or change?",
                           base="Base branch (default: the repo's first allowlisted branch)",
                           branch="Bot branch name (stored as ai/<name>; default: generated)",
                           iter="Continue on an existing bot branch (the named one, else the newest)",
                           title="Pull request title (default: from the prompt)",
                           body="Pull request description (default: the request and the AI's summary)")
    async def repo_task(interaction: discord.Interaction, repo: str,
                        prompt: app_commands.Range[str, 1, MAX_PROMPT_CHARS], base: Optional[str] = None,
                        branch: Optional[str] = None, iter: bool = False,
                        title: Optional[str] = None, body: Optional[str] = None):
        retry = retry_text(repo, prompt, base, branch, iter, title)
        try:
            if interaction.guild_id is None:
                raise rc.ValidationError("This command only works in a server.")
            # Authorize BEFORE deferring (and before the lock): failures are ephemeral and cost nothing.
            rc.authorize(interaction.user.id, interaction.guild_id, mode="target")
            prompt = (prompt or "").strip()
            if not prompt:
                raise rc.ValidationError("Give me something to look into.")
            if len(prompt) > MAX_PROMPT_CHARS:
                raise rc.ValidationError(f"The request is too long ({len(prompt)} chars; max {MAX_PROMPT_CHARS}).")
            spec = rc.resolve_target(repo.strip(), base.strip() if base else None)
            branch_name = bot_branch_name(branch, spec.base) if (branch or "").strip() else None
            title = (title or "").strip() or None
            body = (body or "").strip() or None
            if title and len(title) > MAX_TITLE_CHARS:
                raise rc.ValidationError(f"The title is too long ({len(title)} chars; max {MAX_TITLE_CHARS}).")
            if body and len(body) > MAX_BODY_CHARS:
                raise rc.ValidationError(f"The body is too long ({len(body)} chars; max {MAX_BODY_CHARS}).")
        except rc.RepoBotError as e:
            await _respond_ephemeral(interaction, repotask.format_error(e, retry))
            return
        except Exception:
            log.exception("/repo task failed before it started")
            await _respond_ephemeral(interaction, "Something went wrong before the task could start; "
                                                  "the details are in the bot's log.")
            return

        await interaction.response.defer(thinking=True)          # must happen within 3 s of the command

        task = repotask.TaskContext("target", interaction.user.id, interaction.guild_id,
                                    interaction.channel_id, retry, prompt[:80])

        async def work(_task):
            return await target_job(spec, prompt, client=client_factory(), backend=backend, read_only=False,
                                    branch=branch_name, iterate=iter, title=title, body=body)

        await repotask.run_task(task, make_sender(interaction), work)

    if guild_ids:                                    # visible only where the owner may use it, and syncs instantly
        bot.tree.add_command(group, guilds=[discord.Object(id=g) for g in guild_ids])
    else:
        bot.tree.add_command(group)

    if sync:
        previous = bot.setup_hook

        async def setup_hook():
            await previous()
            await sync_commands(bot, guild_ids)
        bot.setup_hook = setup_hook
    return group