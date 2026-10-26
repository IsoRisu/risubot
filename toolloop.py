# toolloop.py
# The model <-> ToolSession loop (blocking; run it with asyncio.to_thread).
# Shared by the sandbox flow (repocmds) and the target-repo flow (repotarget).
#
#   result = run_tool_loop(session, user_prompt, system_prompt)           # chat=llmtools.chat_tools by default
#   result.text -> the model's final short reply; session holds the staged files.

from __future__ import annotations

import json
from dataclasses import dataclass

import aitools
import llmtools
import repoconfig as rc

MAX_CALLS_PER_ROUND = 8
_BULKY_ARGS = {"write_file": ("content",), "edit_file": ("old_str", "new_str")}
KEEP_RECENT = 2                       # newest messages are never compacted


@dataclass
class LoopResult:
    text: str
    rounds: int
    backend: str = ""


def _fixed_chars(system_prompt: str, tools: list) -> int:
    return len(system_prompt) + len(json.dumps(tools))


def compact_history(messages: list, budget: int, fixed: int) -> int:
    """Shrink OLD bulky content in place until the prompt fits `budget` (oldest first):
    write_file/edit_file arguments become '[N chars omitted]', long tool results are cut.
    The newest KEEP_RECENT messages are untouched. Returns the resulting size."""
    def size():
        return llmtools.history_chars(messages) + fixed

    for m in messages[:-KEEP_RECENT]:
        if size() <= budget:
            break
        if m["role"] == "assistant":
            for c in m.get("calls", []):
                args = c.get("args")
                if not isinstance(args, dict):
                    continue
                for key in _BULKY_ARGS.get(c["name"], ()):
                    v = args.get(key)
                    if isinstance(v, str) and len(v) > 120:
                        args[key] = f"[{len(v)} chars omitted]"
        elif m["role"] == "tool":
            for r in m["results"]:
                if len(r["text"]) > 300:
                    r["text"] = r["text"][:200] + f"\n[... {len(r['text']) - 200} chars omitted to save space ...]"
    return size()


def run_tool_loop(session: aitools.ToolSession, user_prompt: str, system_prompt: str, *,
                  chat=llmtools.chat_tools, budget: int = rc.PROMPT_BUDGET_CHARS) -> LoopResult:
    tools = session.tools()
    fixed = _fixed_chars(system_prompt, tools)
    history: list = [{"role": "user", "text": user_prompt}]
    backend = ""
    while True:
        session.note_round()                                   # raises LimitExceeded past MAX_TOOL_ROUNDS
        compact_history(history, budget, fixed)
        turn = chat(history, tools, system_prompt)
        backend = turn.backend or backend
        if not turn.calls:
            return LoopResult(turn.text, session.rounds, backend)
        history.append({"role": "assistant", "text": turn.text or None, "calls": turn.calls})
        results = []
        for i, call in enumerate(turn.calls):
            if i >= MAX_CALLS_PER_ROUND:
                res = aitools.ToolResult(f"ERROR: too many tool calls in one round (max {MAX_CALLS_PER_ROUND}).", True)
            elif call.get("args") is None:
                res = aitools.ToolResult("ERROR: the arguments were not valid JSON. Call the tool again.", True)
            else:
                res = session.execute(call["name"], call["args"])
            results.append({"id": call["id"], "name": call["name"], "text": res.text, "is_error": res.is_error})
        history.append({"role": "tool", "results": results})
        if session.finished:
            return LoopResult(turn.text, session.rounds, backend)
