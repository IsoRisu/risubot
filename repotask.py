# repotask.py
# One-task-at-a-time lock + chat error/retry messaging for the repo feature.
#
# Usage from a Discord command (Parts 2+):
#   ctx_t = TaskContext(kind="sandbox", user_id=ctx.author.id, guild_id=ctx.guild.id,
#                       channel_id=ctx.channel.id, retry_command="!ask make a snake game")
#   async def work(t):                      # runs while holding the lock
#       ...                                 # blocking calls go through asyncio.to_thread
#       return f"Done: {commit_url}"        # text posted on success (the commit / PR URL)
#   await run_task(ctx_t, ctx.reply, work)  # handles busy / errors / retry hint itself

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import repoconfig as rc
from repoconfig import RepoBotError

log = logging.getLogger("repotask")


class TaskBusy(RepoBotError):
    show_retry = True


@dataclass
class TaskContext:
    kind: str                       # "sandbox" | "target"
    user_id: int
    guild_id: int | None
    channel_id: int | None
    retry_command: str = ""         # exact command the user can paste again; shown in every error message
    summary: str = ""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started: float = field(default_factory=time.monotonic)


class TaskLock:
    """Global single-slot lock. Non-blocking: a second task is rejected, not queued.
    A holder older than max_age seconds is considered hung and can be replaced."""

    def __init__(self, max_age: float = rc.MAX_TASK_SECONDS):
        self.max_age = max_age
        self._mutex = threading.Lock()
        self._current: TaskContext | None = None

    def current(self) -> TaskContext | None:
        with self._mutex:
            return self._current

    def try_acquire(self, task: TaskContext) -> bool:
        with self._mutex:
            cur = self._current
            if cur is not None and time.monotonic() - cur.started <= self.max_age:
                return False
            if cur is not None:
                log.warning("task %s exceeded %ss; lock taken over by %s", cur.task_id, self.max_age, task.task_id)
            task.started = time.monotonic()
            self._current = task
            return True

    def release(self, task: TaskContext):
        with self._mutex:
            if self._current is not None and self._current.task_id == task.task_id:
                self._current = None

    @asynccontextmanager
    async def guard(self, task: TaskContext):
        if not self.try_acquire(task):
            cur = self.current()
            age = int(time.monotonic() - cur.started) if cur else 0
            who = f" by <@{cur.user_id}>" if cur else ""
            raise TaskBusy(f"A repo task is already running (started {age}s ago{who}). "
                           "Only one task runs at a time; wait for it to finish.")
        try:
            yield task
        finally:
            self.release(task)


LOCK = TaskLock()


# ---------------------------------------------------------------------------
# Error -> chat message
# ---------------------------------------------------------------------------
def _describe(exc: Exception) -> tuple:
    """(message, show_retry) for any exception."""
    if isinstance(exc, RepoBotError):
        return exc.user_message, exc.show_retry
    try:                                            # llmclient exceptions, without making it a hard dependency
        import llmclient
        if isinstance(exc, llmclient.LLMKeysRejected):
            return "All AI API keys were rejected. The bot owner needs to check botkey.py.", False
        if isinstance(exc, llmclient.LLMBlocked):
            return f"The AI declined this request ({exc}). Try rephrasing it.", True
        if isinstance(exc, llmclient.LLMUnavailable):
            return str(exc), True                   # already says "try again in N s"
    except ImportError:
        pass
    log.error("unexpected error in repo task", exc_info=exc)
    return f"Unexpected error: {type(exc).__name__}: {str(exc)[:300]}", True


def format_error(exc: Exception, retry_command: str = "") -> str:
    """Chat message (<= 1900 chars, secrets redacted) with the retry command stated in it."""
    msg, show_retry = _describe(exc)
    cmd = (retry_command or "").replace("`", "'").strip()
    retry = ""
    if show_retry and cmd:
        retry = "\nRetry with: `" + (cmd if len(cmd) <= 1000 else cmd[:1000] + "…") + "`"
    text = "❌ " + rc.redact(msg)
    return text[:max(0, rc.DISCORD_MSG_LIMIT - len(retry))] + retry


async def run_task(task: TaskContext, send, work, *, lock: TaskLock = LOCK) -> bool:
    """Hold the lock, run `await work(task)`, post its returned text (commit / PR URL) on success,
    or post the formatted error + retry command. `send` is an async callable taking one str.
    Returns True on success."""
    try:
        async with lock.guard(task):
            result = await work(task)
    except Exception as e:                          # CancelledError is BaseException: it propagates after release
        await send(format_error(e, task.retry_command))
        return False
    if result:
        await send(rc.redact(str(result))[:rc.DISCORD_MSG_LIMIT])
    return True
