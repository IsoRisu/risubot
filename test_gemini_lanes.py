# test_gemini_lanes.py -- offline tests. Run (from the bot folder): python -m unittest test_gemini_lanes -v
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import gemini_lanes as gl

M = "gemini-3.8-flash"
PT = ZoneInfo(gl.QUOTA_TZ)
NOON = datetime(2026, 10, 28, 12, 0, tzinfo=PT).timestamp()          # PDT, far from any reset


class Clock:
    def __init__(self, t=NOON):
        self.t = t

    def __call__(self):
        return self.t


def lanes(*ids):
    return [gl.Lane(i, [f"key-{i}-aaaaaaaa"]) for i in (ids or ("a",))]


def store(ids=("a",), rpd=3, rpm=100, clock=None, path=None):
    return gl.QuotaStore(lanes(*ids), path, rpd={M: rpd}, rpm={M: rpm}, clock=clock or Clock())


class QuotaTests(unittest.TestCase):
    def test_rpd_cap(self):
        s = store(rpd=3)
        self.assertEqual(s.remaining("a", M), 3)
        self.assertEqual([s.reserve("a", M) for _ in range(4)], [True, True, True, False])
        self.assertEqual(s.remaining("a", M), 0)
        self.assertGreater(s.next_free_in("a", M), 3600)      # waits for Pacific midnight

    def test_rpm_window_with_injected_clock(self):
        clock = Clock()
        s = store(rpd=100, rpm=2, clock=clock)
        self.assertTrue(s.reserve("a", M))
        clock.t += 10
        self.assertTrue(s.reserve("a", M))
        self.assertFalse(s.reserve("a", M))
        self.assertAlmostEqual(s.next_free_in("a", M), 50.0)
        clock.t += 49.9                                       # 59.9 s after the first call
        self.assertFalse(s.reserve("a", M))
        clock.t += 0.1                                        # exactly 60 s: first call left the window
        self.assertTrue(s.reserve("a", M))
        self.assertFalse(s.reserve("a", M))
        self.assertEqual(s.remaining("a", M), 100 - 3)        # RPM refusals are not counted

    def test_refused_reserve_counts_nothing(self):
        s = store(rpd=1)
        s.reserve("a", M)
        s.reserve("a", M)
        self.assertEqual(s.remaining("a", M), 0)

    def test_next_free_in_none_when_available(self):
        self.assertIsNone(store().next_free_in("a", M))

    def test_pacific_midnight_reset(self):
        clock = Clock(datetime(2026, 10, 28, 23, 59, 30, tzinfo=PT).timestamp())
        s = store(rpd=2, clock=clock)
        self.assertTrue(s.reserve("a", M))
        self.assertTrue(s.reserve("a", M))
        self.assertFalse(s.reserve("a", M))
        self.assertAlmostEqual(s.next_free_in("a", M), 30.0, places=3)
        clock.t += 31                                         # 00:00:01 Pacific, next day
        self.assertEqual(s.remaining("a", M), 2)
        self.assertTrue(s.reserve("a", M))
        self.assertEqual(s.remaining("a", M), 1)

    def test_reset_follows_pacific_not_utc(self):
        # 00:30 UTC on the 29th is still 17:30 PDT on the 28th: no reset yet.
        clock = Clock(datetime(2026, 10, 28, 17, 0, tzinfo=PT).timestamp())
        s = store(rpd=1, clock=clock)
        s.reserve("a", M)
        clock.t += 1800
        self.assertEqual(s.remaining("a", M), 0)

    def test_unknown_model_and_bad_limits(self):
        with self.assertRaises(ValueError):
            store().reserve("a", "gemini-nope")
        with self.assertRaises(ValueError):
            gl.QuotaStore(lanes(), None, rpd={M: 5}, rpm={M: 0})

    def test_default_limits_match_the_spec(self):
        self.assertEqual(gl.DEFAULT_RPD, {"gemini-3.8-flash": 20, "gemini-3.6-flash": 20, "gemini-3.5-flash-lite": 500})
        self.assertEqual(gl.DEFAULT_RPM, {"gemini-3.8-flash": 5, "gemini-3.6-flash": 5, "gemini-3.5-flash-lite": 15})


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "quota.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        s1 = store(rpd=5, path=self.path)
        s1.reserve("a", M)
        s1.reserve("a", M)
        self.assertEqual(os.listdir(self._tmp.name), ["quota.json"])       # no temp file left behind
        s2 = store(rpd=5, path=self.path)
        self.assertEqual(s2.remaining("a", M), 3)
        s2.reserve("a", M)
        self.assertEqual(store(rpd=5, path=self.path).remaining("a", M), 2)

    def test_old_day_is_discarded(self):
        store(rpd=5, path=self.path).reserve("a", M)
        tomorrow = Clock(NOON + 86400)
        self.assertEqual(store(rpd=5, path=self.path, clock=tomorrow).remaining("a", M), 5)

    def test_missing_file_is_fine(self):
        self.assertEqual(store(rpd=5, path=self.path).remaining("a", M), 5)

    def test_corrupt_file_tolerated(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        with self.assertLogs("gemini_lanes", level="WARNING"):
            s = store(rpd=5, path=self.path)
        self.assertEqual(s.remaining("a", M), 5)
        s.reserve("a", M)                                                   # and it recovers on the next save
        self.assertEqual(json.load(open(self.path))["counts"], {"a": {M: 1}})

    def test_wrong_structure_tolerated(self):
        for bad in ('{"day": "%s", "counts": []}', '{"day": "%s", "counts": {"a": {"%s": "x"}}}', '[1, 2]'):
            day = datetime.fromtimestamp(NOON, PT).date().isoformat()
            with open(self.path, "w") as f:
                f.write(bad.replace("%s", day, 1).replace("%s", M))
            with self.assertLogs("gemini_lanes", level="WARNING"):
                s = store(rpd=5, path=self.path)
            self.assertEqual(s.remaining("a", M), 5)

    def test_unwritable_path_does_not_break_reserve(self):
        s = store(path=os.path.join(self._tmp.name, "no_such_dir", "q.json"))
        with self.assertLogs("gemini_lanes", level="WARNING"):
            self.assertTrue(s.reserve("a", M))


class PickLaneTests(unittest.TestCase):
    def test_most_remaining_wins(self):
        s = store(("a", "b", "c"), rpd=5)
        for _ in range(3):
            s.reserve("a", M)
        s.reserve("b", M)
        self.assertEqual(gl.pick_lane(s, M).id, "c")
        s.reserve("c", M)
        s.reserve("c", M)
        self.assertEqual(gl.pick_lane(s, M).id, "b")

    def test_tie_goes_to_first_lane(self):
        self.assertEqual(gl.pick_lane(store(("a", "b")), M).id, "a")

    def test_skips_lane_without_rpm_headroom(self):
        s = store(("a", "b"), rpd=10, rpm=1)
        s.reserve("b", M)
        s.reserve("a", M)
        s._counts[("a", M)] = 0                                             # a has the most RPD left but is RPM-blocked
        self.assertIsNone(gl.pick_lane(s, M))
        s2 = store(("a", "b"), rpd=10, rpm=1)
        s2.reserve("a", M)
        self.assertEqual(gl.pick_lane(s2, M).id, "b")

    def test_none_when_everything_is_spent(self):
        s = store(("a", "b"), rpd=1)
        s.reserve("a", M)
        s.reserve("b", M)
        self.assertIsNone(gl.pick_lane(s, M))
        self.assertIsNone(gl.acquire(s, M))
        self.assertGreater(gl.min_wait(s, M), 0)

    def test_acquire_reserves_and_balances(self):
        s = store(("a", "b"), rpd=2)
        picked = [gl.acquire(s, M).id for _ in range(4)]
        self.assertEqual(sorted(picked), ["a", "a", "b", "b"])
        self.assertEqual(picked[:2], ["a", "b"])
        self.assertIsNone(gl.acquire(s, M))

    def test_min_wait_none_when_a_lane_is_free(self):
        self.assertIsNone(gl.min_wait(store(("a", "b")), M))


class LoadLanesTests(unittest.TestCase):
    def test_gemini_lanes_preferred(self):
        src = SimpleNamespace(GEMINI_LANES=[["k1", "k2"], ["k3"]], GEMINI_API_KEY=["ignored"])
        got = gl.load_lanes(src)
        self.assertEqual([(l.id, l.keys) for l in got], [("acct1", ["k1", "k2"]), ("acct2", ["k3"])])

    def test_fallback_one_key_per_lane(self):
        got = gl.load_lanes(SimpleNamespace(GEMINI_API_KEY=["k1", "k2"]))
        self.assertEqual([(l.id, l.keys) for l in got], [("acct1", ["k1"]), ("acct2", ["k2"])])
        self.assertEqual([l.keys for l in gl.load_lanes(SimpleNamespace(GEMINI_API_KEY="solo"))], [["solo"]])

    def test_nothing_configured(self):
        self.assertEqual(gl.load_lanes(SimpleNamespace()), [])

    def test_invalid_config(self):
        for bad in ([[]], [["k1"], []], [["k1", ""]], [["k1", 5]], ["k1"], [["k1"], ["k1"]], "k1"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                gl.load_lanes(SimpleNamespace(GEMINI_LANES=bad))

    def test_repr_hides_keys(self):
        self.assertNotIn("secret", repr(gl.Lane("a", ["secret-key"])))

    def test_load_limits_merges_overrides(self):
        rpd, rpm = gl.load_limits(SimpleNamespace(GEMINI_RPD={M: 7}, GEMINI_RPM={"gemini-3.5-flash-lite": 10}))
        self.assertEqual(rpd[M], 7)
        self.assertEqual(rpd["gemini-3.6-flash"], 20)
        self.assertEqual(rpm["gemini-3.5-flash-lite"], 10)
        self.assertEqual(rpm[M], 5)
        with self.assertRaises(ValueError):
            gl.load_limits(SimpleNamespace(GEMINI_RPD=[1]))

    def test_load_limits_reads_repo_limits(self):
        rpd, rpm = gl.load_limits(SimpleNamespace(REPO_LIMITS={"rpd": {M: 9}, "rpm": {M: 2}}))
        self.assertEqual((rpd[M], rpm[M]), (9, 2))
        self.assertEqual(rpd["gemini-3.5-flash-lite"], 500)
        # the flat legacy names win over REPO_LIMITS
        rpd, _ = gl.load_limits(SimpleNamespace(REPO_LIMITS={"rpd": {M: 9}}, GEMINI_RPD={M: 3}))
        self.assertEqual(rpd[M], 3)
        for bad in (SimpleNamespace(REPO_LIMITS=[1]), SimpleNamespace(REPO_LIMITS={"rpd": [1]})):
            with self.assertRaises(ValueError):
                gl.load_limits(bad)


if __name__ == "__main__":
    unittest.main()
