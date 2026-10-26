# repocmds.py
# Discord command for the sandbox flow.  `!code [--branch <name>] <what to build>`
#
# Wiring in risubot.py (two lines, e.g. just above "# --- Run the Bot ---"):
#     import repocmds
#     repocmds.register(bot)
#
# Flow: authorize -> take the task lock -> tool loop (sandbox tools) -> if files were written:
# force-push ONE orphan commit to the sandbox repo -> post the commit URL.
# If the model wrote no files, nothing is pushed and its text reply is posted (the sandbox is not wiped).

from __future__ import annotations

import asyncio
import re

import aitools
import githubclient as gh
import repoconfig as rc
import repotask
import toolloop
import llmtools

MAX_PROMPT_CHARS = 6_000            # user prompt + inlined text attachments
MAX_ATTACH_FILES = 3
MAX_ATTACH_CHARS = 20_000           # per attached text file
MAX_REPLY_CHARS = 900               # model's closing text shown next to the URL

CODE_USAGE = ("Usage: `!code [--branch <name>] <what to build>` — the AI writes files and I push them to the sandbox "
              "repo (it is wiped each time). Text/code files can be attached as input.")

_BRANCH_FLAG = re.compile(r"^\s*--branch(?:=|\s+)(\S+)\s*", re.I)


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
                      chat=llmtools.chat_tools) -> str:
    """The work done while holding the lock. Returns the message to post. Raises on any failure."""
    client = client or gh.GitHubClient()

    def blocking() -> str:
        session = aitools.ToolSession("sandbox")
        result = toolloop.run_tool_loop(session, prompt, aitools.build_tool_system_prompt("sandbox"), chat=chat)
        files = session.changes()
        if not files:
            return result.text or "(The AI produced no files and no reply.)"
        commit = client.sandbox_force_push(files, f"Sandbox output ({len(files)} file(s))", branch)
        return format_success(commit, sorted(files), result.text)

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


def register(bot, *, client_factory=gh.GitHubClient, chat=llmtools.chat_tools):
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
            return await sandbox_job(prompt, branch, client=client_factory(), chat=chat)

        async with ctx.typing():
            await repotask.run_task(task, ctx.reply, work)

    return code_command
