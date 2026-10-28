# gemini_lanes.py
# Account "lanes" and quota bookkeeping for the Gemini-only repo workflow.
#
# A lane is one Google account (one quota pool) with one or more API keys. Quotas are tracked per
# (lane, model):
#   * RPD: requests per day, reset at 00:00 America/Los_Angeles, persisted to a small JSON file
#   * RPM: sliding 60-second window, in memory only
#
#   lanes = load_lanes()                                  # from botkey
#   rpd, rpm = load_limits()
#   store = QuotaStore(lanes, "gemini_quota.json", rpd=rpd, rpm=rpm)
#   lane = acquire(store, "gemini-3.8-flash")             # pick_lane + reserve; None = nothing free
#   if lane is None:
#       wait = min_wait(store, "gemini-3.8-flash")        # seconds until some lane frees up (or None)
#
# botkey.py variables (all optional except one of the first two):
#   GEMINI_LANES   = [["AIza..a1", "AIza..a2"], ["AIza..b1"]]   # one inner list per Google account
#   GEMINI_API_KEY = ["AIza..", "AIza.."]                       # fallback: one key = one lane
#   REPO_LIMITS    = {"rpd": {"gemini-3.8-flash": 20},          # overrides merged over DEFAULT_RPD
#                     "rpm": {"gemini-3.8-flash": 5}}           # ... and DEFAULT_RPM (GEMINI_RPD / GEMINI_RPM also work)
#
# Semantics worth knowing:
#   * next_free_in() returns None when the (lane, model) can be used right now, otherwise the
#     seconds to wait (the larger of the RPM wait and, if RPD is used up, the time to Pacific midnight).
#   * The RPM window is not persisted: right after a restart a lane may burst up to RPM calls.
#   * pick_lane() only looks; reserve() is the atomic step. Use acquire() to do both.

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

log = logging.getLogger("gemini_lanes")

QUOTA_TZ = "America/Los_Angeles"
RPM_WINDOW_SECONDS = 60.0

# Defaults, overridable from botkey (see load_limits). Not enforced anywhere else.
DEFAULT_RPD = {"gemini-3.8-flash": 20, "gemini-3.6-flash": 20, "gemini-3.5-flash-lite": 500}
DEFAULT_RPM = {"gemini-3.8-flash": 5, "gemini-3.6-flash": 5, "gemini-3.5-flash-lite": 15}


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------
@dataclass
class Lane:
    id: str
    keys: list = field(default_factory=list)

    def __repr__(self) -> str:                       # never print keys
        return f"Lane(id={self.id!r}, keys=<{len(self.keys)}>)"


def _clean_keys(keys, where: str) -> list:
    if isinstance(keys, str) or not isinstance(keys, (list, tuple)):
        raise ValueError(f"{where} must be a list of key strings.")
    out = []
    for k in keys:
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"{where} contains an empty or non-string key.")
        out.append(k.strip())
    if not out:
        raise ValueError(f"{where} is empty.")
    return list(dict.fromkeys(out))                  # drop duplicates inside one lane, keep order


def load_lanes(source=None) -> list:
    """Lanes from botkey.GEMINI_LANES (list of key lists, one per account), else from
    botkey.GEMINI_API_KEY with one key per lane. `source` (any object with those attributes)
    replaces botkey in tests. Raises ValueError on bad config; returns [] when nothing is set."""
    if source is None:
        import botkey as source
    raw = getattr(source, "GEMINI_LANES", None)
    lanes = []
    if raw:
        if not isinstance(raw, (list, tuple)):
            raise ValueError("GEMINI_LANES must be a list of lists of keys.")
        for i, keys in enumerate(raw, 1):
            lanes.append(Lane(f"acct{i}", _clean_keys(keys, f"GEMINI_LANES[{i - 1}]")))
    else:
        single = getattr(source, "GEMINI_API_KEY", None)
        if isinstance(single, str):
            single = [single]
        if single:
            for i, key in enumerate(_clean_keys(single, "GEMINI_API_KEY"), 1):
                lanes.append(Lane(f"acct{i}", [key]))
    seen = {}
    for lane in lanes:                               # one key in two lanes would count one quota twice
        for k in lane.keys:
            if k in seen:
                raise ValueError(f"The same API key is in two lanes ({seen[k]} and {lane.id}).")
            seen[k] = lane.id
    return lanes


def load_limits(source=None) -> tuple:
    """(rpd, rpm) dicts: the defaults, overridden per model by botkey.REPO_LIMITS = {"rpd": {...}, "rpm": {...}}.
    The older flat names botkey.GEMINI_RPD / GEMINI_RPM still work and win over REPO_LIMITS if both are set."""
    if source is None:
        import botkey as source
    rpd, rpm = dict(DEFAULT_RPD), dict(DEFAULT_RPM)
    limits = getattr(source, "REPO_LIMITS", None)
    if limits is not None and not isinstance(limits, dict):
        raise ValueError("REPO_LIMITS must be a dict like {\"rpd\": {model: limit}, \"rpm\": {model: limit}}.")
    limits = limits or {}
    for target, key, flat in ((rpd, "rpd", "GEMINI_RPD"), (rpm, "rpm", "GEMINI_RPM")):
        for name, extra in ((f"REPO_LIMITS[{key!r}]", limits.get(key)), (flat, getattr(source, flat, None))):
            if extra:
                if not isinstance(extra, dict):
                    raise ValueError(f"{name} must be a dict {{model: limit}}.")
                target.update(extra)
    return rpd, rpm


# ---------------------------------------------------------------------------
# Quota store
# ---------------------------------------------------------------------------
def _pacific_tz():
    try:
        return ZoneInfo(QUOTA_TZ)
    except Exception:                                # no tz database (e.g. Windows without tzdata)
        log.warning("time zone %s unavailable; using fixed UTC-8 (install tzdata for exact resets)", QUOTA_TZ)
        return timezone(timedelta(hours=-8))


def _lane_id(lane) -> str:
    return lane if isinstance(lane, str) else lane.id


class QuotaStore:
    """Daily + per-minute counters per (lane, model). Thread-safe. `clock` returns epoch seconds
    (injectable for tests)."""

    def __init__(self, lanes, path: str | None = None, *, rpd: dict | None = None, rpm: dict | None = None,
                 clock=time.time):
        self.lanes = list(lanes)
        self.path = path
        self.rpd = dict(DEFAULT_RPD if rpd is None else rpd)
        self.rpm = dict(DEFAULT_RPM if rpm is None else rpm)
        for model, v in self.rpd.items():
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"RPD for {model} must be an integer >= 0.")
        for model, v in self.rpm.items():
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"RPM for {model} must be an integer >= 1.")
        self._clock = clock
        self._tz = _pacific_tz()
        self._lock = threading.RLock()
        self._counts: dict = {}                      # (lane_id, model) -> calls today
        self._windows: dict = {}                     # (lane_id, model) -> deque of timestamps
        self._day = self._pacific_date(self._clock())
        self._load()

    # -- time ------------------------------------------------------------
    def _pacific_date(self, now: float) -> str:
        return datetime.fromtimestamp(now, self._tz).date().isoformat()

    def _secs_to_reset(self, now: float) -> float:
        local = datetime.fromtimestamp(now, self._tz)
        nxt = datetime.combine(local.date() + timedelta(days=1), dtime(0), tzinfo=self._tz)
        return max(0.0, nxt.timestamp() - now)

    def _roll(self, now: float):
        day = self._pacific_date(now)
        if day != self._day:
            log.info("quota day rolled over: %s -> %s", self._day, day)
            self._day = day
            self._counts.clear()

    # -- limits ------------------------------------------------------------
    def _limits(self, model: str) -> tuple:
        if model not in self.rpd or model not in self.rpm:
            raise ValueError(f"No RPD/RPM configured for model {model!r}.")
        return self.rpd[model], self.rpm[model]

    def _wait(self, lid: str, model: str, now: float) -> float:
        """Seconds until (lane, model) can take a call; 0.0 = now. Caller holds the lock."""
        rpd, rpm = self._limits(model)
        key = (lid, model)
        waits = []
        if self._counts.get(key, 0) >= rpd:
            waits.append(self._secs_to_reset(now))
        win = self._windows.get(key)
        if win:
            while win and win[0] <= now - RPM_WINDOW_SECONDS:
                win.popleft()
            if len(win) >= rpm:
                waits.append(win[0] + RPM_WINDOW_SECONDS - now)
        return max(waits) if waits else 0.0

    # -- public API ----------------------------------------------------------
    def reserve(self, lane, model: str) -> bool:
        """Atomically check RPD and RPM; on success count the call and return True."""
        lid = _lane_id(lane)
        with self._lock:
            now = self._clock()
            self._roll(now)
            if self._wait(lid, model, now) > 0:
                return False
            key = (lid, model)
            self._counts[key] = self._counts.get(key, 0) + 1
            self._windows.setdefault(key, deque()).append(now)
            self._save()
            return True

    def can_reserve(self, lane, model: str) -> bool:
        """True when reserve() would succeed right now (does not count anything)."""
        with self._lock:
            now = self._clock()
            self._roll(now)
            return self._wait(_lane_id(lane), model, now) == 0.0

    def remaining(self, lane, model: str) -> int:
        """Calls left today for (lane, model), ignoring RPM."""
        with self._lock:
            self._roll(self._clock())
            rpd, _ = self._limits(model)
            return max(0, rpd - self._counts.get((_lane_id(lane), model), 0))

    def next_free_in(self, lane, model: str):
        """None when usable now, else seconds to wait (RPM window, or Pacific midnight if RPD is spent)."""
        with self._lock:
            now = self._clock()
            self._roll(now)
            wait = self._wait(_lane_id(lane), model, now)
            return wait if wait > 0 else None

    # -- persistence ----------------------------------------------------------
    def _save(self):
        if not self.path:
            return
        counts: dict = {}
        for (lid, model), n in self._counts.items():
            counts.setdefault(lid, {})[model] = n
        payload = json.dumps({"version": 1, "day": self._day, "counts": counts})
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self.path)               # atomic: a crash can't leave a half-written file
        except OSError as e:
            log.warning("could not save quota counters to %s: %s: %s", self.path, type(e).__name__, e)

    def _load(self):
        if not self.path:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as e:
            log.warning("quota file %s is unreadable (%s); starting with empty counters", self.path, e)
            return
        try:
            if data["day"] != self._day:
                return                               # counters from an earlier Pacific day: discard
            loaded = {}
            for lid, per_model in data["counts"].items():
                for model, n in per_model.items():
                    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                        raise ValueError(f"bad count {n!r}")
                    loaded[(str(lid), str(model))] = n
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            log.warning("quota file %s has an unexpected structure (%s); starting with empty counters",
                        self.path, e)
            return
        self._counts = loaded


# ---------------------------------------------------------------------------
# Lane selection
# ---------------------------------------------------------------------------
def pick_lane(store: QuotaStore, model: str):
    """The lane with the most remaining RPD among those that have RPM headroom now (ties: config
    order). None when no lane can take a call. Does not reserve; see acquire()."""
    best, best_left = None, 0
    for lane in store.lanes:
        if not store.can_reserve(lane, model):
            continue
        left = store.remaining(lane, model)
        if left > best_left:
            best, best_left = lane, left
    return best


def acquire(store: QuotaStore, model: str):
    """pick_lane + reserve, retried if another thread takes the slot in between. None = nothing free."""
    for _ in range(len(store.lanes) + 1):
        lane = pick_lane(store, model)
        if lane is None:
            return None
        if store.reserve(lane, model):
            return lane
    return None


def min_wait(store: QuotaStore, model: str):
    """Seconds until the first lane can take a call for `model`, or None if one can now / no lanes."""
    waits = []
    for lane in store.lanes:
        w = store.next_free_in(lane, model)
        if w is None:
            return None
        waits.append(w)
    return min(waits) if waits else None
