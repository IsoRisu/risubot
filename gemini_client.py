# gemini_client.py
# One blocking Gemini generateContent call with lane rotation (standard library only).
#
#   reply = call_gemini(store, lanes, "gemini-3.8-flash", system_prompt, user_text)
#   reply.text, reply.lane_id, reply.model, reply.finish_reason
#
# Flow per call: pick a lane (gemini_lanes.pick_lane) -> store.reserve() -> HTTP request. The
# reservation is made BEFORE EVERY request, including the one retry on the same lane.
#   200       text extracted from candidates[0].content.parts (parts flagged "thought" are skipped)
#   429       lane is cooled down for this model (RetryInfo.retryDelay, else Retry-After, else 60 s)
#             and the next lane is tried
#   401/403   (and 400 API_KEY_INVALID) that key is disabled for the process; a lane whose keys are all
#             disabled is out of rotation
#   5xx/408/timeout/network error   retried once on the same lane, then the next lane
#   other 4xx the request itself is bad: GeminiUnavailable(permanent=True), no rotation
# No lane can answer -> GeminiUnavailable(retry_after=seconds or None). Content refused -> GeminiBlocked.
# Every attempt is logged as "lane=<id> model=<m> status=<s>" (never the key).
#
# ASSUMPTIONS about the Gemini REST API (verify against current docs before relying on them):
#   * POST .../v1beta/models/{model}:generateContent with header x-goog-api-key.
#   * Body: systemInstruction.parts[].text, contents[{role, parts[{text}]}],
#     generationConfig.maxOutputTokens / responseMimeType.
#   * Success: candidates[0].content.parts[].text and candidates[0].finishReason; a refused prompt has
#     promptFeedback.blockReason and no candidates.
#   * 429 body: error.details[] containing an entry whose "@type" ends in RetryInfo with
#     "retryDelay": "34s" (protobuf Duration as a string). 400 for a bad key carries API_KEY_INVALID.
#
# Failure state (cooldowns, disabled keys) lives in LaneHealth, not in gemini_lanes.QuotaStore, so
# Part 1 is untouched. The module-level HEALTH is the default; tests pass their own.

from __future__ import annotations

import http.client
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from gemini_lanes import pick_lane

log = logging.getLogger("gemini_client")

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_COOLDOWN = 60.0            # 429 without a retry hint
MIN_COOLDOWN = 1.0
MAX_COOLDOWN = 24 * 3600.0
RETRY_PAUSE = 1.0                  # pause before the one same-lane retry after a 5xx / timeout
BLOCK_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION",
                                  "IMAGE_SAFETY"})


# ---------------------------------------------------------------------------
# Errors and result
# ---------------------------------------------------------------------------
class GeminiUnavailable(RuntimeError):
    """No lane could answer (quota, cooldown, outage, rejected keys) or the request was refused as malformed.
    retry_after: seconds until some lane is expected to be usable, or None when unknown.
    permanent: retrying will not help (bad request, every key rejected)."""

    def __init__(self, msg, retry_after=None, permanent=False):
        super().__init__(msg)
        self.retry_after = retry_after
        self.permanent = permanent


class GeminiBlocked(RuntimeError):
    """Gemini refused the prompt or the answer (safety / policy). Another lane will not help."""


@dataclass
class GeminiReply:
    text: str
    lane_id: str
    model: str
    finish_reason: str = ""        # e.g. "STOP" or "MAX_TOKENS" (a cut-off answer); "" when absent


# ---------------------------------------------------------------------------
# Lane health: cooldowns after 429, keys disabled after 401/403
# ---------------------------------------------------------------------------
class LaneHealth:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._cool: dict = {}                      # (lane_id, model) -> clock time the cooldown ends
        self._dead: set = set()                    # (lane_id, key)

    def cool(self, lane_id: str, model: str, secs: float):
        with self._lock:
            self._cool[(lane_id, model)] = self._clock() + secs

    def cooling_for(self, lane_id: str, model: str) -> float:
        """Seconds of cooldown left (0.0 when none)."""
        with self._lock:
            return max(0.0, self._cool.get((lane_id, model), 0.0) - self._clock())

    def disable_key(self, lane_id: str, key: str):
        with self._lock:
            self._dead.add((lane_id, key))

    def live_key(self, lane):
        with self._lock:
            for k in lane.keys:
                if (lane.id, k) not in self._dead:
                    return k
        return None

    def lane_disabled(self, lane) -> bool:
        return self.live_key(lane) is None


HEALTH = LaneHealth()


# ---------------------------------------------------------------------------
# Transport (injectable): (url, headers, body_bytes, timeout) -> (status, headers_dict, body_bytes)
# It returns HTTP error statuses as values; network failures and timeouts are raised (OSError family).
# ---------------------------------------------------------------------------
def urllib_transport(url: str, headers: dict, body: bytes, timeout: float):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as e:
        try:
            data = e.read()
        finally:
            e.close()
        return e.code, dict(e.headers.items()) if e.headers else {}, data


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------
def build_body(system_prompt: str, user_text: str, *, json_mode: bool, max_output: int) -> bytes:
    gen = {"maxOutputTokens": int(max_output)}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    return json.dumps({
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": gen,
    }).encode("utf-8")


_DURATION = re.compile(r"(\d+(?:\.\d+)?)s")


def _json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return None


def retry_delay(headers: dict, raw: bytes) -> float:
    """Cooldown for a 429: RetryInfo.retryDelay from the body, else Retry-After, else DEFAULT_COOLDOWN."""
    delay = None
    data = _json(raw)
    err = data.get("error") if isinstance(data, dict) else None
    for d in (err.get("details") if isinstance(err, dict) else None) or []:
        if isinstance(d, dict) and str(d.get("@type", "")).endswith("RetryInfo"):
            m = _DURATION.fullmatch(str(d.get("retryDelay", "")).strip())
            if m:
                delay = float(m.group(1))
                break
    if delay is None:
        lowered = {str(k).lower(): v for k, v in (headers or {}).items()}
        try:
            if lowered.get("retry-after"):
                delay = float(lowered["retry-after"])
        except (TypeError, ValueError):
            pass
    if delay is None:
        delay = DEFAULT_COOLDOWN
    return min(MAX_COOLDOWN, max(MIN_COOLDOWN, delay))


def _key_invalid(raw: bytes) -> bool:
    low = raw.decode("utf-8", "replace").lower()
    return any(s in low for s in ("api_key_invalid", "api key not valid", "api key expired"))


def _scrub(text: str, lanes) -> str:
    for lane in lanes:
        for k in lane.keys:
            if len(k) >= 4:
                text = text.replace(k, "***")
    return " ".join(text.split())[:200]


def _parse(raw: bytes, lane_id: str, model: str) -> GeminiReply:
    payload = _json(raw)
    if not isinstance(payload, dict):
        raise GeminiUnavailable(f"gemini/{model}: the response was not valid JSON")
    cands = payload.get("candidates") or []
    if not cands:
        reason = (payload.get("promptFeedback") or {}).get("blockReason")
        if reason:
            raise GeminiBlocked(f"Gemini blocked the request ({reason}).")
        raise GeminiUnavailable(f"gemini/{model}: response had no candidates")
    cand = cands[0]
    finish = str(cand.get("finishReason") or "")
    if finish in BLOCK_FINISH_REASONS:
        raise GeminiBlocked(f"Gemini declined to answer ({finish}).")
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p["text"] for p in parts
                   if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought")).strip()
    if not text:
        raise GeminiUnavailable(f"gemini/{model}: empty response (finishReason={finish or 'none'})")
    return GeminiReply(text, lane_id, model, finish)


# ---------------------------------------------------------------------------
# Lane choice
# ---------------------------------------------------------------------------
class _EligibleView:
    """What pick_lane needs from a store (.lanes, can_reserve, remaining), limited to eligible lanes."""

    def __init__(self, store, lanes):
        self._store = store
        self.lanes = lanes

    def can_reserve(self, lane, model):
        return self._store.can_reserve(lane, model)

    def remaining(self, lane, model):
        return self._store.remaining(lane, model)


def _choose(store, lanes, model, health, failed):
    eligible = [l for l in lanes
                if l.id not in failed and not health.lane_disabled(l) and health.cooling_for(l.id, model) <= 0]
    return pick_lane(_EligibleView(store, eligible), model) if eligible else None


def _nothing_left(store, lanes, model, health) -> GeminiUnavailable:
    live = [l for l in lanes if not health.lane_disabled(l)]
    if not live:
        return GeminiUnavailable("All Gemini API keys were rejected (or none are configured). "
                                 "The bot owner needs to check botkey.py.", permanent=True)
    waits = []
    for lane in live:
        w = max(health.cooling_for(lane.id, model), store.next_free_in(lane, model) or 0.0)
        if w > 0:
            waits.append(w)
    retry = min(waits) if waits else None
    hint = f" Try again in about {int(retry) + 1} s." if retry else " Please try again in a minute."
    return GeminiUnavailable(f"The Gemini accounts are out of quota or unavailable for {model}." + hint,
                             retry_after=retry)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------
def _try_lane(store, lanes, lane, model, body, health, transport, timeout, sleep, failed):
    """Up to two requests on one lane. Returns a GeminiReply, or None to move on to the next lane."""
    key = health.live_key(lane)
    if key is None:
        failed.add(lane.id)
        return None
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    url = API_URL.format(model=model)
    for attempt in (1, 2):
        if not store.reserve(lane, model):                 # lost a race / RPM window filled up
            failed.add(lane.id)
            return None
        try:
            status, resp_headers, raw = transport(url, headers, body, timeout)
        except (OSError, http.client.HTTPException) as e:  # TimeoutError and URLError are OSErrors
            log.warning("gemini call lane=%s model=%s status=%s", lane.id, model, type(e).__name__)
        else:
            log.info("gemini call lane=%s model=%s status=%s", lane.id, model, status)
            if status == 200:
                return _parse(raw, lane.id, model)
            if status == 429:
                secs = retry_delay(resp_headers, raw)
                health.cool(lane.id, model, secs)
                failed.add(lane.id)
                log.warning("lane=%s model=%s rate limited, cooling %.0f s", lane.id, model, secs)
                return None
            if status in (401, 403) or (status == 400 and _key_invalid(raw)):
                health.disable_key(lane.id, key)
                log.error("lane=%s: an API key was rejected (HTTP %s) and is disabled until restart", lane.id, status)
                return None
            if not (status >= 500 or status == 408):
                raise GeminiUnavailable(f"Gemini rejected the request (HTTP {status}): "
                                        f"{_scrub(raw.decode('utf-8', 'replace'), lanes)}", permanent=True)
        if attempt == 1:                                   # 5xx / timeout / network error: one retry here
            sleep(RETRY_PAUSE)
    failed.add(lane.id)
    return None


def call_gemini(store, lanes, model: str, system_prompt: str, user_text: str, *, json_mode: bool = True,
                max_output: int = 4000, timeout: float = 60, transport=None, health=None,
                sleep=time.sleep) -> GeminiReply:
    """One Gemini answer. Blocking. `transport`, `health` and `sleep` are test hooks.
    Raises GeminiUnavailable / GeminiBlocked."""
    transport = transport or urllib_transport
    health = health if health is not None else HEALTH
    body = build_body(system_prompt, user_text, json_mode=json_mode, max_output=max_output)
    failed: set = set()                                    # lanes given up on during THIS call
    while True:
        lane = _choose(store, lanes, model, health, failed)
        if lane is None:
            raise _nothing_left(store, lanes, model, health)
        reply = _try_lane(store, lanes, lane, model, body, health, transport, timeout, sleep, failed)
        if reply is not None:
            return reply
