# llmtools.py
# Function-calling on top of llmclient's providers, keys, cooldowns and fallback.
# llmclient.py itself is NOT modified; this module reuses its provider objects
# (llmclient._providers, _KeyPool, _handle_http_error) so quota state is shared with !ask.
#
# Provider-neutral history (what callers keep and what survives a provider switch mid-task):
#   {"role": "user",      "text": str}
#   {"role": "assistant", "text": str | None, "calls": [{"id", "name", "args": dict | None, "_sig": str | None}]}
#   {"role": "tool",      "results": [{"id", "name", "text", "is_error"}]}      # one message per round
#
#   turn = chat_tools(history, tools_specs, system_prompt)   # blocking; -> ToolTurn(text, calls, backend)
#   args is None when the model produced unparseable arguments (the loop reports that back to the model).

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import uuid
from dataclasses import dataclass, field
import http.client

import llmclient
import repoconfig as rc

log = logging.getLogger("llmtools")

# Gemini 3 validates "thought signatures" on function-call turns. Calls that came from another
# provider have none; Google documents this placeholder to skip validation. VERIFY against current docs.
GEMINI_SKIP_SIGNATURE = "skip_thought_signature_validator"


@dataclass
class ToolTurn:
    text: str
    calls: list = field(default_factory=list)
    backend: str = ""


def new_call_id() -> str:
    """9 alphanumerics: the strictest format any provider accepts (Mistral)."""
    return uuid.uuid4().hex[:9]


def history_chars(messages: list) -> int:
    """Rough prompt size of a neutral history (what counts against provider max_chars)."""
    n = 0
    for m in messages:
        if m["role"] == "user":
            n += len(m["text"])
        elif m["role"] == "assistant":
            n += len(m.get("text") or "")
            n += sum(len(c["name"]) + len(json.dumps(c.get("args") or {}, ensure_ascii=False)) + 30 for c in m.get("calls", []))
        else:
            n += sum(len(r["text"]) + len(r["name"]) + 30 for r in m["results"])
    return n


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
def _openai_messages(messages: list, system_prompt: str) -> list:
    out = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("text") or None}
            if m.get("calls"):
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"],
                                                   "arguments": json.dumps(c.get("args") or {}, ensure_ascii=False)}}
                                     for c in m["calls"]]
            out.append(msg)
        else:
            for r in m["results"]:
                out.append({"role": "tool", "tool_call_id": r["id"], "content": r["text"]})
    return out


def _gemini_contents(messages: list, model: str) -> list:
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "parts": [{"text": m["text"]}]})
        elif m["role"] == "assistant":
            parts = [{"text": m["text"]}] if m.get("text") else []
            for c in m.get("calls", []):
                part = {"functionCall": {"name": c["name"], "args": c.get("args") or {}}}
                sig = c.get("_sig") or (GEMINI_SKIP_SIGNATURE if model.startswith("gemini-3") else None)
                if sig:
                    part["thoughtSignature"] = sig
                parts.append(part)
            out.append({"role": "model", "parts": parts or [{"text": "(no output)"}]})
        else:
            out.append({"role": "user", "parts": [
                {"functionResponse": {"name": r["name"], "response": {"error" if r["is_error"] else "result": r["text"]}}}
                for r in m["results"]]})
    return out


def build_body(p, model: str, messages: list, tools: list, system_prompt: str) -> bytes:
    if isinstance(p, llmclient._GeminiProvider):
        body = {"systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": _gemini_contents(messages, model),
                "tools": [{"functionDeclarations": tools}],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}}}
    else:
        body = {"model": model, "messages": _openai_messages(messages, system_prompt),
                "tools": [{"type": "function", "function": t} for t in tools], "tool_choice": "auto"}
        body.update(p.extra_body)
    return json.dumps(body).encode("utf-8")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_response(p, payload: dict, model: str) -> ToolTurn:
    backend = f"{p.name}/{model}"
    if isinstance(p, llmclient._GeminiProvider):
        cands = payload.get("candidates") or []
        if not cands:
            reason = (payload.get("promptFeedback") or {}).get("blockReason")
            if reason:
                raise llmclient.LLMBlocked(f"Gemini blocked the request ({reason}).")
            raise llmclient.LLMUnavailable(f"{backend}: response had no candidates")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(x["text"] for x in parts if "text" in x and not x.get("thought")).strip()
        calls = []
        for x in parts:
            fc = x.get("functionCall")
            if fc and fc.get("name"):
                args = fc.get("args")
                calls.append({"id": new_call_id(), "name": fc["name"],
                              "args": args if isinstance(args, dict) else {}, "_sig": x.get("thoughtSignature")})
        if not text and not calls:
            reason = cands[0].get("finishReason")
            if reason in llmclient.BLOCK_FINISH_REASONS:
                raise llmclient.LLMBlocked(f"Gemini declined to answer ({reason}).")
            raise llmclient.LLMUnavailable(f"{backend}: empty response (finishReason={reason})")
        return ToolTurn(text, calls, backend)

    choices = payload.get("choices") or []
    if not choices:
        raise llmclient.LLMUnavailable(f"{backend}: no choices")
    choice = choices[0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        content = "".join(x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") in (None, "text"))
    text = llmclient._THINK_RE.sub("", content or "").strip()
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if not fn.get("name"):
            continue
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            args = raw
        else:
            try:
                args = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                args = None
        calls.append({"id": new_call_id(), "name": fn["name"], "args": args if isinstance(args, dict) else None, "_sig": None})
    if not text and not calls:
        reason = choice.get("finish_reason")
        if reason == "content_filter":
            raise llmclient.LLMBlocked("The model declined to answer (content_filter).")
        raise llmclient.LLMUnavailable(f"{backend}: empty response (finish_reason={reason})")
    return ToolTurn(text, calls, backend)


# ---------------------------------------------------------------------------
# One model, all keys (mirrors llmclient._Provider.call: same cooldown bookkeeping)
# ---------------------------------------------------------------------------
def _call_model(p, model: str, body: bytes, deadline: float) -> ToolTurn:
    label = f"{p.name}/{model}"
    order, soonest = p.pool.plan(model)
    if not order:
        raise llmclient.LLMUnavailable(f"{label}: every key is cooling down", retry_after=soonest)
    timeout = min(p.request_timeout, llmclient.TEXT_REQUEST_TIMEOUT)
    last = ""
    for idx in order:
        for attempt in range(1, llmclient.ATTEMPTS_PER_KEY + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                raise llmclient.LLMUnavailable(f"{label}: out of time ({last})")
            try:
                payload = p._post(model, idx, body, min(timeout, remaining))
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")[:2000]
                last = f"key#{idx + 1} HTTP {e.code}: {llmclient._brief(text, 120)}"
                kind = llmclient._handle_http_error(p, idx, model, e.code, text, e.headers)
                if kind == "rotate":
                    break
                if kind == "model_missing":
                    raise llmclient.LLMUnavailable(f"{label}: not found")
                if kind == "skip":
                    raise llmclient.LLMUnavailable(f"{label}: {last}", permanent=True)
            except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as e:
                last = f"key#{idx + 1} {type(e).__name__}: {llmclient._brief(str(e), 120)}"
                if llmclient._is_timeout(e):
                    p.pool.cool_model(model, llmclient.MODEL_TIMEOUT_COOLDOWN)
                    raise llmclient.LLMUnavailable(f"{label}: {last}")
            else:
                p.pool.stick(idx)
                return parse_response(p, payload, model)
            if attempt < llmclient.ATTEMPTS_PER_KEY:
                time.sleep(max(0.0, min(random.uniform(0.7, 1.3) * 1.5 * attempt, deadline - time.monotonic() - 1)))
        else:
            p.pool.cool_model(model, llmclient.MODEL_OVERLOAD_COOLDOWN)
            raise llmclient.LLMUnavailable(f"{label}: {last}")
    raise llmclient.LLMUnavailable(f"{label}: no key could answer ({last})", retry_after=p.pool.soonest_wait())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def chat_tools(messages: list, tools: list, system_prompt: str) -> ToolTurn:
    """One model round. Walks providers (in llmclient order) and their models until one answers.
    Providers whose free-tier prompt limit is smaller than the history are skipped (clipping would corrupt it).
    Raises llmclient.LLMUnavailable / LLMBlocked / LLMKeysRejected, repoconfig.LimitExceeded, RuntimeError."""
    size = history_chars(messages) + len(system_prompt) + len(json.dumps(tools))
    plan = [p for p in llmclient._providers if size <= p.max_chars]
    if not plan:
        raise rc.LimitExceeded(f"The task context grew too large ({size} chars) for the free AI providers. "
                               "Try a smaller request.")
    deadline = time.monotonic() + llmclient.TOTAL_DEADLINE
    errors, waits, perms = [], [], []
    for n, p in enumerate(plan):
        now = time.monotonic()
        if now >= deadline:
            errors.append("deadline reached")
            perms.append(False)
            break
        slice_end = min(deadline, now + max(llmclient.MIN_PROVIDER_SLICE, (deadline - now) / (len(plan) - n)))
        tried = 0
        for model in p.candidates(False):
            if time.monotonic() >= slice_end or tried >= llmclient.MAX_MODELS_PER_PROVIDER:
                break
            tried += 1
            try:
                turn = _call_model(p, model, build_body(p, model, messages, tools, system_prompt), slice_end)
            except llmclient.LLMUnavailable as e:
                errors.append(str(e))
                perms.append(e.permanent)
                if e.retry_after is not None:
                    waits.append(e.retry_after)
                continue
            llmclient._tls.backend = turn.backend
            log.info("tool round answered by %s", turn.backend)
            return turn
    if all(p.pool.all_rejected() for p in llmclient._providers):
        raise llmclient.LLMKeysRejected("All API keys of all providers were rejected as invalid. Check botkey.py.")
    if errors and all(perms):
        raise RuntimeError("Every provider rejected the request: " + " | ".join(errors)[:600])
    log.warning("all providers unavailable (tools): %s", " | ".join(errors))
    soonest = [w for w in (p.pool.soonest_wait() for p in llmclient._providers) if w is not None]
    retry = min(waits) if waits else (min(soonest) if soonest else None)
    hint = f" Try again in about {int(retry) + 1} s." if retry else " Please try again in a minute."
    raise llmclient.LLMUnavailable("The AI is overloaded or out of quota right now." + hint, retry_after=retry)
