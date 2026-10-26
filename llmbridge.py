# llmbridge.py
# Option A bridge: the LOCAL Ollama model runs every tool round; a single remote
# call (llmclient.ask with a small prompt) writes the final Discord reply. Keeps
# free remote quota for what only a remote provider does well: the polished text.
#
# Wiring (see repocmds.sandbox_job and repotarget.run_target):
#     result = llmbridge.run_bridged_loop(session, prompt, system_prompt, local_chat=chat)
#
# Failure handling:
#   * remote finalize raises LLMUnavailable / LLMKeysRejected -> we return the local
#     model's text prefixed with a ⚠️ Remote AI unavailable (<reason>) marker, so the
#     user still gets an answer.
#   * local tool loop raises anything (LLMUnavailable, LLMBlocked, rc.LimitExceeded,
#     rc.ValidationError, ...) -> propagates untouched; repotask.format_error formats it.
#
# Injectable hooks (so tests never hit the network):
#   local_chat        (messages, tools, system_prompt) -> llmtools.ToolTurn
#                     default: llmtools.chat_tools_local
#   remote_finalize   (prompt, local_text, changes) -> str
#                     default: one llmclient.ask() with FINALIZE_SYSTEM_PROMPT

from __future__ import annotations

import llmclient
import llmtools
import repoconfig as rc
import toolloop


# Small, Discord-friendly prompt for the single remote call that writes the reply.
# Deliberately NOT llmclient.SYSTEM_PROMPT (~11k): this prompt uses little of the
# free-tier quota and its job is narrow (reformat + polish the answer).
FINALIZE_SYSTEM_PROMPT = r"""You are writing the final reply to the user in a Discord chat.
The work has already been done by another AI; you are the editor who turns its notes into the reply.

Reply in the same language the user wrote in (Finnish if they wrote Finnish).
Be concise. Use Discord Markdown (**bold**, *italic*, short headings, numbered lists).
Math: inline in $...$, display in $$...$$. Code: in ``` fences with a language tag.
Never mention the other AI, this message, the prompt, or that you are summarizing: write as if you did the work.
No meta-commentary, no disclaimers, no follow-up questions. If files were changed, mention them briefly."""


def _default_remote_finalize(prompt: str, local_text: str, changes: dict) -> str:
    """Build a compact finalize prompt from (user prompt, local summary, staged files)
    and send it through llmclient.ask with FINALIZE_SYSTEM_PROMPT."""
    if changes:
        names = sorted(changes)
        shown = ", ".join(f"`{n}`" for n in names[:20])
        if len(names) > 20:
            shown += f" (+{len(names) - 20} more)"
        files_line = f"{len(names)} file(s): {shown}"
    else:
        files_line = "none (the AI did not write or edit any files)"
    final_prompt = (
        "The user's request:\n"
        f"{prompt}\n\n"
        "What the AI did (its own summary, may be terse or empty):\n"
        f"{local_text or '(no text summary)'}\n\n"
        f"Files changed: {files_line}\n\n"
        "Write the final reply to the user now."
    )
    return llmclient.ask(final_prompt, system_prompt=FINALIZE_SYSTEM_PROMPT)


def run_bridged_loop(session, prompt: str, system_prompt: str, *,
                     local_chat=None,
                     remote_finalize=None,
                     budget: int = rc.PROMPT_BUDGET_CHARS) -> toolloop.LoopResult:
    """Run the tool loop on the LOCAL model, then one remote call for the reply.

    * local_chat defaults to llmtools.chat_tools_local (single-provider, Ollama).
      Tests can pass a scripted fake with the same signature.
    * remote_finalize defaults to _default_remote_finalize; tests can pass a fake
      that records its args and returns a fixed string.
    * rc.LimitExceeded, rc.ValidationError and llmclient.LLMBlocked from either
      side propagate unchanged so repotask.format_error() renders them normally.

    Returns a LoopResult whose .text is the remote reply, .rounds is the local
    round count, and .backend is "local/model → remote/provider" (or just the
    local backend when the remote finalize failed and we fell back to the local
    text).
    """
    local_chat = local_chat or llmtools.chat_tools_local
    remote_finalize = remote_finalize or _default_remote_finalize

    result = toolloop.run_tool_loop(session, prompt, system_prompt, chat=local_chat, budget=budget)

    try:
        final_text = remote_finalize(prompt, result.text, session.changes())
    except (llmclient.LLMUnavailable, llmclient.LLMKeysRejected) as e:
        # Normalize the reason to a short, user-visible string. LLMKeysRejected is
        # rare here (llmclient already retries every configured key); showing the
        # raw exception text in the marker keeps the message honest without being
        # too long.
        if isinstance(e, llmclient.LLMKeysRejected):
            reason = "keys rejected"
        else:
            reason = str(e) or "unavailable"
        marker = f"⚠️ Remote AI unavailable ({reason})"
        body = result.text or "(no reply)"
        return toolloop.LoopResult(marker + "\n\n" + body, result.rounds, result.backend)

    backend = f"{result.backend} → {llmclient.last_backend()}"
    return toolloop.LoopResult(final_text or "", result.rounds, backend)