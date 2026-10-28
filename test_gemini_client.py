# test_gemini_client.py -- offline tests. Run (from the bot folder): python -m unittest test_gemini_client -v
import io
import json
import unittest
import urllib.error
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import gemini_client as gc
import gemini_lanes as gl

M = "gemini-3.8-flash"
NOON = datetime(2026, 10, 28, 12, 0, tzinfo=ZoneInfo(gl.QUOTA_TZ)).timestamp()


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def ok(text='{"a": 1}', finish="STOP", parts=None):
    parts = parts if parts is not None else [{"text": text}]
    body = {"candidates": [{"content": {"parts": parts}, "finishReason": finish}]}
    return 200, {}, json.dumps(body).encode()


def err(code, message="problem", delay=None, headers=None):
    error = {"code": code, "message": message}
    if delay:
        error["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]
    return code, headers or {}, json.dumps({"error": error}).encode()


class SpyStore(gl.QuotaStore):
    """Real QuotaStore that records every reserve() in a shared event log."""

    def __init__(self, lanes, path, *, events, **kw):
        self.events = events
        super().__init__(lanes, path, **kw)

    def reserve(self, lane, model):
        got = super().reserve(lane, model)
        self.events.append(("reserve", lane if isinstance(lane, str) else lane.id, got))
        return got


class Transport:
    """Scripted transport: script maps an API key to a list of responses (tuples) or exceptions."""

    def __init__(self, script, events):
        self.script = {k: list(v) for k, v in script.items()}
        self.events, self.calls = events, []

    def __call__(self, url, headers, body, timeout):
        key = headers["x-goog-api-key"]
        self.events.append(("request", key))
        self.calls.append(SimpleNamespace(url=url, headers=headers, body=json.loads(body), timeout=timeout))
        queue = self.script.get(key)
        if not queue:
            raise AssertionError(f"unexpected request with {key}")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def assert_reserved_before_each_request(test, events):
    """Every request must directly follow a successful reservation of the lane that owns the key."""
    pending = None
    for ev in events:
        if ev[0] == "reserve":
            pending = ev[1] if ev[2] else None
        else:
            test.assertIsNotNone(pending, f"request without a reservation: {events}")
            test.assertTrue(ev[1].startswith(f"key-{pending}-"), f"request for the wrong lane: {events}")
            pending = None


class Base(unittest.TestCase):
    def setUp(self):
        self.events, self.sleeps = [], []
        self.hclock = Clock(1000.0)
        self.health = gc.LaneHealth(clock=self.hclock)

    def make(self, ids=("a", "b"), script=None, keys=None, rpd=5, rpm=100):
        keys = keys or {}
        self.lanes = [gl.Lane(i, keys.get(i, [f"key-{i}-1"])) for i in ids]
        self.store = SpyStore(self.lanes, None, events=self.events, rpd={M: rpd}, rpm={M: rpm}, clock=Clock(NOON))
        self.transport = Transport(script or {}, self.events)

    def call(self, **kw):
        return gc.call_gemini(self.store, self.lanes, M, "SYS", "hi", transport=self.transport,
                              health=self.health, sleep=self.sleeps.append, **kw)

    def requested_keys(self):
        return [e[1] for e in self.events if e[0] == "request"]


class RequestShapeTests(Base):
    def test_request_and_reply(self):
        self.make(("a",), {"key-a-1": [ok('{"x": 1}')]})
        reply = self.call(timeout=7)
        self.assertEqual((reply.text, reply.lane_id, reply.model, reply.finish_reason), ('{"x": 1}', "a", M, "STOP"))
        c, = self.transport.calls
        self.assertEqual(c.url, f"https://generativelanguage.googleapis.com/v1beta/models/{M}:generateContent")
        self.assertEqual(c.headers["x-goog-api-key"], "key-a-1")
        self.assertEqual(c.timeout, 7)
        self.assertEqual(c.body["systemInstruction"], {"parts": [{"text": "SYS"}]})
        self.assertEqual(c.body["contents"], [{"role": "user", "parts": [{"text": "hi"}]}])
        self.assertEqual(c.body["generationConfig"], {"maxOutputTokens": 4000, "responseMimeType": "application/json"})
        self.assertEqual(self.store.remaining("a", M), 4)

    def test_plain_text_mode_and_max_output(self):
        self.make(("a",), {"key-a-1": [ok("hello")]})
        self.call(json_mode=False, max_output=123)
        self.assertEqual(self.transport.calls[0].body["generationConfig"], {"maxOutputTokens": 123})

    def test_thought_parts_are_skipped_and_truncation_is_visible(self):
        self.make(("a",), {"key-a-1": [ok(parts=[{"text": "thinking", "thought": True}, {"text": "{\"a\":"}],
                                          finish="MAX_TOKENS")]})
        reply = self.call()
        self.assertEqual((reply.text, reply.finish_reason), ('{"a":', "MAX_TOKENS"))


class RotationTests(Base):
    def test_429_rotates_and_the_lane_stays_cooled(self):
        self.make(script={"key-a-1": [err(429, delay="34s"), ok()], "key-b-1": [ok(), ok()]})
        self.store.reserve("b", M)                                   # a: 5 left, b: 4 left -> a is picked first
        self.events.clear()
        reply = self.call()
        self.assertEqual(reply.lane_id, "b")
        self.assertEqual(self.requested_keys(), ["key-a-1", "key-b-1"])
        self.assertAlmostEqual(self.health.cooling_for("a", M), 34.0)
        self.events.clear()
        self.assertEqual(self.call().lane_id, "b")                   # a has more quota left but is cooling
        self.assertEqual(self.requested_keys(), ["key-b-1"])
        self.hclock.t += 35                                          # cooldown over: a (4 left) beats b (2 left)
        self.events.clear()
        self.assertEqual(self.call().lane_id, "a")
        self.assertEqual(self.requested_keys(), ["key-a-1"])

    def test_429_default_cooldown_and_retry_after_header(self):
        self.make(("a", "b"), {"key-a-1": [err(429)], "key-b-1": [err(429, headers={"Retry-After": "12"})]})
        with self.assertRaises(gc.GeminiUnavailable):
            self.call()
        self.assertAlmostEqual(self.health.cooling_for("a", M), 60.0)
        self.assertAlmostEqual(self.health.cooling_for("b", M), 12.0)

    def test_401_disables_the_key_for_good(self):
        for status, body in ((401, "x"), (403, "x"), (400, "API key not valid. API_KEY_INVALID")):
            with self.subTest(status=status):
                self.events.clear()
                self.health = gc.LaneHealth(clock=self.hclock)
                self.make(script={"key-a-1": [err(status, body)], "key-b-1": [ok(), ok()]})
                self.assertEqual(self.call().lane_id, "b")
                self.assertTrue(self.health.lane_disabled(self.lanes[0]))
                self.assertEqual(self.call().lane_id, "b")           # a is never contacted again (no script left)
                self.assertEqual(self.requested_keys().count("key-a-1"), 1)

    def test_a_lane_with_a_second_key_keeps_working(self):
        self.make(("a",), {"key-a-1": [err(401)], "key-a-2": [ok(), ok()]},
                  keys={"a": ["key-a-1", "key-a-2"]})
        self.assertEqual(self.call().lane_id, "a")
        self.assertEqual(self.requested_keys(), ["key-a-1", "key-a-2"])
        self.assertFalse(self.health.lane_disabled(self.lanes[0]))
        self.events.clear()
        self.call()
        self.assertEqual(self.requested_keys(), ["key-a-2"])

    def test_every_key_rejected_is_permanent(self):
        self.make(("a", "b"), {"key-a-1": [err(401)], "key-b-1": [err(403)]})
        with self.assertRaises(gc.GeminiUnavailable) as cm:
            self.call()
        self.assertTrue(cm.exception.permanent)
        self.assertIsNone(cm.exception.retry_after)
        with self.assertRaises(gc.GeminiUnavailable):                # and nothing is sent afterwards
            self.call()
        self.assertEqual(len(self.transport.calls), 2)


class RetryTests(Base):
    def test_503_is_retried_on_the_same_lane(self):
        self.make(script={"key-a-1": [err(503), ok()]})
        self.assertEqual(self.call().lane_id, "a")
        self.assertEqual(self.requested_keys(), ["key-a-1", "key-a-1"])
        self.assertEqual(self.sleeps, [gc.RETRY_PAUSE])
        self.assertEqual(self.store.remaining("a", M), 3)            # the retry is a counted request too

    def test_timeout_is_retried_like_a_5xx(self):
        self.make(script={"key-a-1": [TimeoutError("timed out"), ok()]})
        self.assertEqual(self.call().lane_id, "a")
        self.assertEqual(self.requested_keys(), ["key-a-1", "key-a-1"])

    def test_network_error_moves_on_after_one_retry(self):
        self.make(script={"key-a-1": [urllib.error.URLError("down"), ConnectionResetError()], "key-b-1": [ok()]})
        self.assertEqual(self.call().lane_id, "b")
        self.assertEqual(self.requested_keys(), ["key-a-1", "key-a-1", "key-b-1"])
        self.assertEqual(self.sleeps, [gc.RETRY_PAUSE])

    def test_two_failures_then_next_lane_without_cooling(self):
        self.make(script={"key-a-1": [err(503), err(500)], "key-b-1": [ok()]})
        self.assertEqual(self.call().lane_id, "b")
        self.assertEqual(self.health.cooling_for("a", M), 0.0)


class UnavailableTests(Base):
    def test_all_lanes_429_gives_the_earliest_retry_time(self):
        self.make(script={"key-a-1": [err(429, delay="45s")], "key-b-1": [err(429, delay="30s")]})
        with self.assertRaises(gc.GeminiUnavailable) as cm:
            self.call()
        self.assertAlmostEqual(cm.exception.retry_after, 30.0)
        self.assertFalse(cm.exception.permanent)
        self.assertIn("31 s", str(cm.exception))

    def test_exhausted_quota_sends_nothing_and_reports_the_reset(self):
        self.make(rpd=1)
        self.store.reserve("a", M)
        self.store.reserve("b", M)
        self.events.clear()
        with self.assertRaises(gc.GeminiUnavailable) as cm:
            self.call()
        self.assertEqual(self.events, [])
        self.assertGreater(cm.exception.retry_after, 0)
        self.assertLessEqual(cm.exception.retry_after, 24 * 3600)

    def test_cooldown_and_exhausted_quota_use_the_longer_wait(self):
        self.make(("a",), rpd=1, script={"key-a-1": [ok()]})
        self.call()
        self.health.cool("a", M, 10)
        with self.assertRaises(gc.GeminiUnavailable) as cm:
            self.call()
        self.assertGreater(cm.exception.retry_after, 10)             # midnight is further away than the cooldown

    def test_only_server_errors_leave_the_retry_time_unknown(self):
        self.make(script={"key-a-1": [err(503), err(503)], "key-b-1": [err(500), err(500)]})
        with self.assertRaises(gc.GeminiUnavailable) as cm:
            self.call()
        self.assertIsNone(cm.exception.retry_after)
        self.assertFalse(cm.exception.permanent)
        self.assertEqual(len(self.requested_keys()), 4)

    def test_no_lanes_at_all(self):
        self.make(ids=())
        with self.assertRaises(gc.GeminiUnavailable) as cm:
            self.call()
        self.assertTrue(cm.exception.permanent)

    def test_bad_request_is_permanent_and_does_not_rotate_or_leak_the_key(self):
        self.make(script={"key-a-1": [err(400, "bad field, key-a-1 was sent")], "key-b-1": [ok()]})
        with self.assertRaises(gc.GeminiUnavailable) as cm:
            self.call()
        self.assertTrue(cm.exception.permanent)
        self.assertNotIn("key-a-1", str(cm.exception))
        self.assertEqual(self.requested_keys(), ["key-a-1"])

    def test_empty_or_unparseable_answers(self):
        for resp in (ok(parts=[]), ok(parts=[{"text": "   "}]), (200, {}, b'{"candidates": []}'),
                     (200, {}, b"<html>not json</html>")):
            with self.subTest(resp=resp[2][:30]):
                self.events.clear()
                self.make(script={"key-a-1": [resp], "key-b-1": [ok()]})
                with self.assertRaises(gc.GeminiUnavailable):
                    self.call()
                self.assertEqual(self.requested_keys(), ["key-a-1"])


class BlockedTests(Base):
    def test_finish_reason_safety(self):
        for reason in ("SAFETY", "PROHIBITED_CONTENT", "RECITATION"):
            with self.subTest(reason=reason):
                self.events.clear()
                self.make(script={"key-a-1": [ok(parts=[], finish=reason)], "key-b-1": [ok()]})
                with self.assertRaises(gc.GeminiBlocked) as cm:
                    self.call()
                self.assertIn(reason, str(cm.exception))
                self.assertEqual(self.requested_keys(), ["key-a-1"])    # no rotation: another lane would refuse too

    def test_safety_wins_even_with_partial_text(self):
        self.make(script={"key-a-1": [ok("half an ans", finish="SAFETY")]})
        with self.assertRaises(gc.GeminiBlocked):
            self.call()

    def test_prompt_feedback_block(self):
        body = json.dumps({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}).encode()
        self.make(script={"key-a-1": [(200, {}, body)]})
        with self.assertRaises(gc.GeminiBlocked):
            self.call()


class ReservationTests(Base):
    def test_a_reservation_precedes_every_request(self):
        self.make(script={"key-a-1": [err(503), err(503)], "key-b-1": [err(429, delay="5s")]},
                  keys={})
        with self.assertRaises(gc.GeminiUnavailable):
            self.call()
        assert_reserved_before_each_request(self, self.events)
        self.assertEqual(len(self.requested_keys()), 3)
        self.assertEqual(self.store.remaining("a", M), 3)
        self.assertEqual(self.store.remaining("b", M), 4)

    def test_rpm_refusal_on_the_retry_moves_to_the_next_lane_without_a_request(self):
        self.make(rpm=1, script={"key-a-1": [err(503)], "key-b-1": [ok()]})
        self.assertEqual(self.call().lane_id, "b")                    # a's retry cannot be reserved (RPM 1)
        self.assertEqual(self.requested_keys(), ["key-a-1", "key-b-1"])
        self.assertIn(("reserve", "a", False), self.events)
        assert_reserved_before_each_request(self, self.events)

    def test_the_success_path_obeys_the_ordering_too(self):
        self.make(script={"key-a-1": [err(429)], "key-b-1": [ok()]})
        self.call()
        assert_reserved_before_each_request(self, self.events)


class UrllibTransportTests(unittest.TestCase):
    def test_success_and_http_error_become_values(self):
        class Resp:
            status = 200
            headers = {"X": "y"}

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with patch("urllib.request.urlopen", return_value=Resp()) as m:
            self.assertEqual(gc.urllib_transport("http://x", {"k": "v"}, b"b", 3), (200, {"X": "y"}, b"{}"))
            self.assertEqual(m.call_args.kwargs["timeout"], 3)
        boom = urllib.error.HTTPError("http://x", 429, "no", {"Retry-After": "5"}, io.BytesIO(b'{"e":1}'))
        with patch("urllib.request.urlopen", side_effect=boom):
            self.assertEqual(gc.urllib_transport("http://x", {}, b"b", 3), (429, {"Retry-After": "5"}, b'{"e":1}'))

    def test_timeouts_propagate(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            with self.assertRaises(TimeoutError):
                gc.urllib_transport("http://x", {}, b"b", 1)


class RetryDelayTests(unittest.TestCase):
    def test_sources_and_clamping(self):
        self.assertEqual(gc.retry_delay({}, err(429, delay="34.5s")[2]), 34.5)
        self.assertEqual(gc.retry_delay({"retry-after": "9"}, b"{}"), 9.0)
        self.assertEqual(gc.retry_delay({"Retry-After": "soon"}, b"not json"), gc.DEFAULT_COOLDOWN)
        self.assertEqual(gc.retry_delay({}, err(429, delay="0s")[2]), gc.MIN_COOLDOWN)
        self.assertEqual(gc.retry_delay({}, err(429, delay="999999s")[2]), gc.MAX_COOLDOWN)


if __name__ == "__main__":
    unittest.main()
