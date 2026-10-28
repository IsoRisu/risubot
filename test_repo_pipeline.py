# test_repo_pipeline.py -- offline tests. Run (from the bot folder): python -m unittest test_repo_pipeline -v
# call_gemini is replaced by a scripted fake; no network, no real keys, no GitHub.
import json
import sys
import types
import unittest
from unittest.mock import patch

if "botkey" not in sys.modules:                       # same stub idea as test_repo_commands.py
    bk = types.ModuleType("botkey")
    bk.bot_key = "DISCORDSECRET.aaaaaa.bbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    bk.GITHUB_TOKEN = "github_pat_" + "A" * 50
    bk.REPO_OWNER_IDS = [42]
    bk.REPO_ALLOWED_GUILD_IDS = [7]
    bk.SANDBOX_REPO = "me/sandbox"
    bk.REPO_ALLOWLIST = {"me/app": ["main"]}
    sys.modules["botkey"] = bk

import gemini_client as gc  # noqa: E402
import repo_pipeline as rp  # noqa: E402
import repoconfig as rc  # noqa: E402

APP = "def f():\n    return 1\n"
FILES = {"src/app.py": APP, "src/util.py": "X = 1\n"}


def J(**d):
    return json.dumps(d)


def plan(*paths, text="change f"):
    return J(need_files=list(paths), plan=text)


def edits(*es, summary="Done.", **extra):
    return J(summary=summary, edits=list(es), **extra)


def rep(path, search, replace):
    return {"path": path, "op": "replace", "search": search, "replace": replace}


def create(path, content):
    return {"path": path, "op": "create", "replace": content}


GOOD_EDIT = edits(rep("src/app.py", "return 1", "return 2"))


class Script:
    """Fake call_gemini. items: str | (text, lane_id) | Exception | callable(call) -> one of those.
    handler(call) may be given instead of items. Every call is recorded in .calls."""

    def __init__(self, items=None, handler=None):
        self.items = iter(items or [])
        self.handler = handler
        self.calls = []

    def __call__(self, store, lanes, model, system_prompt, user_text, **kw):
        call = dict(model=model, system=system_prompt, user=user_text, kw=kw)
        self.calls.append(call)
        item = self.handler(call) if self.handler else next(self.items)
        if callable(item):
            item = item(call)
        if isinstance(item, Exception):
            raise item
        text, lane = item if isinstance(item, tuple) else (item, "acct1")
        return gc.GeminiReply(text, lane, model, "STOP")

    def models(self):
        return [c["model"] for c in self.calls]


def reader(files=None):
    files = dict(FILES if files is None else files)
    seen = []

    def read_file(path):
        seen.append(path)
        return files.get(path)
    read_file.seen = seen
    return read_file


def run(script, files=None, read_file=None, request="make f return 2", tree="src/app.py\nsrc/util.py", notes="README"):
    return rp.run_task(request, tree, notes, read_file or reader(files), object(), [], call=script)


def unavailable(retry_after=100.0, permanent=False):
    return gc.GeminiUnavailable("out of quota", retry_after=retry_after, permanent=permanent)


class HappyPathTests(unittest.TestCase):
    def test_plan_edit_commit_ready_result(self):
        script = Script([plan("src/app.py"),
                         ("" + edits(rep("src/app.py", "return 1", "return 2"), create("src/new.py", "Y = 2\n"),
                                     summary="Changed f and added new.py."), "acct2")])
        rf = reader()
        res = run(script, read_file=rf)
        self.assertEqual(res.summary, "Changed f and added new.py.")
        self.assertEqual(res.changes, {"src/app.py": "def f():\n    return 2\n", "src/new.py": "Y = 2\n"})
        self.assertEqual(script.models(), [rp.PLAN_MODEL, rp.EDIT_MODEL])
        self.assertEqual([(c.stage, c.lane, c.model, c.status) for c in res.calls],
                         [("plan", "acct1", rp.PLAN_MODEL, "ok"), ("edit", "acct2", rp.EDIT_MODEL, "ok")])
        plan_call, edit_call = script.calls
        self.assertIs(plan_call["system"], rp.PLAN_SYSTEM)
        self.assertIs(edit_call["system"], rp.EDIT_SYSTEM)
        self.assertTrue(plan_call["kw"]["json_mode"])
        self.assertEqual(edit_call["kw"]["max_output"], 4000)
        for needle in ("make f return 2", "src/app.py\nsrc/util.py", "README"):
            self.assertIn(needle, plan_call["user"])
        self.assertIn("=== FILE: src/app.py", edit_call["user"])
        self.assertIn("return 1", edit_call["user"])
        self.assertIn("change f", edit_call["user"])
        self.assertNotIn("X = 1", edit_call["user"])                  # util.py was not requested
        self.assertEqual(rf.seen, ["src/app.py", "src/new.py"])       # the 2nd read is the "does it exist" check for a create

    def test_no_edits_means_empty_changes(self):
        res = run(Script([plan(), edits(summary="Nothing to do.")]))
        self.assertEqual((res.changes, res.summary), ({}, "Nothing to do."))
        self.assertEqual(len(res.calls), 2)

    def test_crlf_file_is_restored_in_changes(self):
        files = {"a.py": "x = 1\r\ny = 2\r\n"}
        res = run(Script([plan("a.py"), edits(rep("a.py", "y = 2", "y = 3"))]), files=files)
        self.assertEqual(res.changes, {"a.py": "x = 1\r\ny = 3\r\n"})
        self.assertEqual(res.normalized, ["a.py"])

    def test_errors_are_typed_repo_errors(self):
        for cls in (rp.QuotaExhausted, rp.AIUnavailable, rp.AIDeclined, rp.AIOutputError, rp.EditsFailed,
                    rp.TaskTooLarge):
            self.assertTrue(issubclass(cls, rc.RepoBotError))
            self.assertTrue(cls.show_retry)


class MalformedJsonTests(unittest.TestCase):
    def test_malformed_plan_is_retried_once_with_the_error(self):
        script = Script(["this is not json", plan("src/app.py"), GOOD_EDIT])
        res = run(script)
        self.assertEqual(script.models(), [rp.PLAN_MODEL, rp.PLAN_MODEL, rp.EDIT_MODEL])
        retry = script.calls[1]["user"]
        self.assertIn("YOUR PREVIOUS REPLY WAS REJECTED", retry)
        self.assertIn("not valid JSON", retry)
        self.assertIn("this is not json", retry)
        self.assertIn("make f return 2", retry)                       # the original prompt is resent
        self.assertEqual([c.status for c in res.calls], ["malformed_json", "ok", "ok"])

    def test_two_malformed_plans_fail_the_task(self):
        script = Script(["nope", "{still nope"])
        with self.assertRaises(rp.AIOutputError) as cm:
            run(script)
        self.assertEqual(len(script.calls), 2)
        self.assertTrue(cm.exception.show_retry)
        self.assertIn("Nothing was changed", str(cm.exception))

    def test_plan_schema_errors_are_malformed_too(self):
        for bad in ('{"need_files": "a.py", "plan": "p"}', '{"need_files": [1], "plan": "p"}',
                    '{"need_files": []}', "[]", ""):
            with self.assertRaises(rp.PlanParseError, msg=bad):
                rp.parse_plan_reply(bad)

    def test_malformed_edit_reply_retried_with_parse_error(self):
        bad = edits(rep("../evil.py", "a", "b"))                       # path rejected by workspace.parse_edit_reply
        script = Script([plan("src/app.py"), bad, GOOD_EDIT])
        res = run(script)
        self.assertIn("edits[0].path", script.calls[2]["user"])
        self.assertEqual(res.changes, {"src/app.py": "def f():\n    return 2\n"})
        self.assertEqual([c.status for c in res.calls], ["ok", "malformed_json", "ok"])

    def test_truncated_edit_reply_gets_a_hint(self):
        def handler(call):
            if call["system"] is rp.PLAN_SYSTEM:
                return plan("src/app.py")
            if "REJECTED" in call["user"]:
                return GOOD_EDIT
            return lambda c: gc.GeminiReply('{"summary": "x", "edits": [', "acct1", c["model"], "MAX_TOKENS")
        script = Script(handler=handler)

        def fake(store, lanes, model, system, user, **kw):
            item = handler(dict(model=model, system=system, user=user))
            script.calls.append(dict(model=model, system=system, user=user, kw=kw))
            return item(dict(model=model)) if callable(item) else gc.GeminiReply(item, "acct1", model, "STOP")
        run(fake)
        self.assertIn("cut off at the output limit", script.calls[2]["user"])
        self.assertIn('"continue": true', script.calls[2]["user"])

    def test_second_malformed_edit_reply_fails(self):
        script = Script([plan("src/app.py"), "x", "y"])
        with self.assertRaises(rp.AIOutputError):
            run(script)
        self.assertEqual(len(script.calls), 3)


class RepairTests(unittest.TestCase):
    BAD = edits(rep("src/app.py", "return 7", "return 2"))            # anchor does not exist

    def test_apply_failure_is_repaired_once(self):
        script = Script([plan("src/app.py"), self.BAD, edits(rep("src/app.py", "return 1", "return 2"), summary="fix")])
        res = run(script)
        self.assertEqual(res.changes, {"src/app.py": "def f():\n    return 2\n"})
        self.assertEqual(script.models(), [rp.PLAN_MODEL, rp.EDIT_MODEL, rp.REPAIR_MODEL])
        repair = script.calls[2]
        self.assertIs(repair["system"], rp.REPAIR_SYSTEM)
        self.assertIn("return 7", repair["user"])                      # the failed edit
        self.assertIn("not found", repair["user"])                     # and its reason
        self.assertIn("=== CURRENT FILE: src/app.py ===", repair["user"])
        self.assertIn("return 1", repair["user"])                      # current text of the file
        self.assertNotIn("make f return 2", repair["user"])            # only failed edits + files
        self.assertEqual([c.stage for c in res.calls], ["plan", "edit", "repair"])
        self.assertEqual(res.calls[2].model, rp.REPAIR_MODEL)

    def test_second_apply_failure_fails_the_task(self):
        still_bad = edits(rep("src/app.py", "return 9", "return 2"))
        script = Script([plan("src/app.py"), self.BAD, still_bad])
        with self.assertRaises(rp.EditsFailed) as cm:
            run(script)
        self.assertEqual(len(script.calls), 3)                         # exactly one repair round, no more calls
        self.assertIn("src/app.py", str(cm.exception))
        self.assertIn("Nothing was committed", str(cm.exception))

    def test_repair_that_ignores_the_failure_does_not_pass(self):
        script = Script([plan("src/app.py"), self.BAD, edits(summary="gave up")])
        with self.assertRaises(rp.EditsFailed):
            run(script)

    def test_partial_failure_keeps_good_edits(self):
        mixed = edits(rep("src/app.py", "return 1", "return 2"), rep("src/util.py", "Y = 1", "Y = 2"))
        script = Script([plan("src/app.py", "src/util.py"), mixed, edits(rep("src/util.py", "X = 1", "X = 2"))])
        res = run(script)
        self.assertEqual(res.changes, {"src/app.py": "def f():\n    return 2\n", "src/util.py": "X = 2\n"})
        self.assertNotIn("=== CURRENT FILE: src/app.py", script.calls[2]["user"])

    def test_edit_on_a_file_that_was_never_shown_skips_the_repair_call(self):
        script = Script([plan("src/app.py"), edits(rep("src/app.py", "return 1", "return 2"),
                                                   {"path": "src/ghost.py", "op": "delete"})])
        with self.assertRaises(rp.EditsFailed):
            run(script)
        self.assertEqual(len(script.calls), 2)                         # unfixable: no repair call

    def test_delete_of_an_already_missing_file_counts_as_done(self):
        # src/util.py is requested, deleted by the first edit; the model deletes it again -> goal already met
        first = edits({"path": "src/util.py", "op": "delete"}, {"path": "src/util.py", "op": "delete"})
        script = Script([plan("src/util.py"), first, edits(summary="nothing left to do")])
        res = run(script)
        self.assertEqual(res.changes, {"src/util.py": None})
        self.assertEqual(script.models()[-1], rp.REPAIR_MODEL)

    def test_verification_error_triggers_repair(self):
        broken = edits(rep("src/app.py", "return 1", "return (1"))
        script = Script([plan("src/app.py"), broken, edits(rep("src/app.py", "return (1", "return 2"))])
        res = run(script)
        self.assertEqual(res.changes, {"src/app.py": "def f():\n    return 2\n"})
        self.assertIn("SyntaxError", script.calls[2]["user"])

    def test_verification_error_after_repair_fails(self):
        broken = edits(rep("src/app.py", "return 1", "return (1"))
        script = Script([plan("src/app.py"), broken, edits(rep("src/app.py", "return (1", "return (2"))])
        with self.assertRaises(rp.EditsFailed) as cm:
            run(script)
        self.assertIn("SyntaxError", str(cm.exception))
        self.assertEqual(len(script.calls), 3)

    def test_broken_json_file_is_caught(self):
        files = {"c.json": '{"a": 1}\n'}
        script = Script([plan("c.json"), edits(rep("c.json", '"a": 1', '"a": 1,')), edits(rep("c.json", '"a": 1,', '"a": 2'))])
        res = run(script, files=files)
        self.assertEqual(res.changes, {"c.json": '{"a": 2}\n'})

    def test_repair_malformed_json_retried_once_then_fails(self):
        script = Script([plan("src/app.py"), self.BAD, "junk", "junk2"])
        with self.assertRaises(rp.AIOutputError):
            run(script)
        self.assertEqual(script.models()[-2:], [rp.REPAIR_MODEL, rp.REPAIR_MODEL])


class ContinuationTests(unittest.TestCase):
    def test_continuation_passes_the_applied_state(self):
        first = edits(rep("src/app.py", "return 1", "return 2"), summary="Part 1.", **{"continue": True})
        second = edits(create("src/new.py", "Z = 3\n"), summary="Part 2.")
        script = Script([plan("src/app.py"), first, second])
        res = run(script)
        self.assertEqual(res.changes, {"src/app.py": "def f():\n    return 2\n", "src/new.py": "Z = 3\n"})
        self.assertEqual(res.summary, "Part 1. Part 2.")
        again = script.calls[2]["user"]
        self.assertIn("CONTINUATION 1", again)
        self.assertIn("return 2", again)                              # state AFTER the first edits
        self.assertNotIn("return 1", again)
        self.assertIn("- replace `src/app.py`", again)
        self.assertEqual(script.models(), [rp.PLAN_MODEL, rp.EDIT_MODEL, rp.EDIT_MODEL])

    def test_continuation_loop_is_bounded(self):
        more = edits(summary="more", **{"continue": True})
        script = Script(handler=lambda c: plan("src/app.py") if c["system"] is rp.PLAN_SYSTEM else more)
        with self.assertRaises(rp.TaskTooLarge) as cm:
            run(script)
        self.assertEqual(len(script.calls), 1 + 1 + rp.MAX_EDIT_CONTINUATIONS)       # plan + first edit + continuations
        self.assertEqual([c["system"] is rp.EDIT_SYSTEM for c in script.calls], [False, True, True, True])
        self.assertIn("Nothing was committed", str(cm.exception))

    def test_failed_edit_in_round_one_is_reported_to_round_two(self):
        first = edits(rep("src/app.py", "return 7", "x"), summary="a", **{"continue": True})
        second = edits(summary="b")
        script = Script([plan("src/app.py"), first, second, edits(rep("src/app.py", "return 1", "return 2"))])
        res = run(script)
        self.assertIn("FAILED (a repair step handles these later", script.calls[2]["user"])
        self.assertEqual(res.changes, {"src/app.py": "def f():\n    return 2\n"})


class FileBudgetTests(unittest.TestCase):
    def test_oversized_file_is_refused_and_never_truncated(self):
        big = "A" * 70
        with patch.object(rp, "MAX_FILE_CHARS_PER_READ", 50):
            script = Script([plan("big.txt", "src/app.py"), edits(rep("src/app.py", "return 1", "return 2"))])
            res = run(script, files={"big.txt": big, "src/app.py": APP})
        edit_prompt = script.calls[1]["user"]
        self.assertNotIn("AAAA", edit_prompt)
        self.assertIn("`big.txt`", edit_prompt)
        self.assertIn("NOT shown", edit_prompt)
        self.assertIn("70 chars, over the 50-char per-file limit", edit_prompt)
        self.assertEqual(res.notes[0].split(":")[0], "- `big.txt`")
        self.assertIn("src/app.py", res.changes)

    def test_total_budget_is_enforced(self):
        files = {"a.txt": "a" * 60, "b.txt": "b" * 60, "c.txt": "c" * 30}
        with patch.object(rp, "FILE_BUDGET_CHARS", 100):
            script = Script([plan("a.txt", "b.txt", "c.txt"), edits(summary="ok")])
            run(script, files=files)
        prompt = script.calls[1]["user"]
        self.assertIn("a" * 60, prompt)
        self.assertNotIn("b" * 10, prompt)                            # refused whole, not cut
        self.assertIn("c" * 30, prompt)                               # a later, smaller file still fits
        self.assertIn("total file budget", prompt)

    def test_editing_a_refused_file_fails_without_a_repair_call(self):
        with patch.object(rp, "MAX_FILE_CHARS_PER_READ", 5):
            script = Script([plan("src/app.py"), edits(rep("src/app.py", "return 1", "return 2"))])
            with self.assertRaises(rp.EditsFailed) as cm:
                run(script)
        self.assertEqual(len(script.calls), 2)
        self.assertIn("not provided to the editor", str(cm.exception))

    def test_create_over_an_unseen_existing_file_is_refused(self):
        script = Script([plan("src/app.py"), edits(create("src/util.py", "X = 99\n"))])
        with self.assertRaises(rp.EditsFailed) as cm:
            run(script)
        self.assertEqual(len(script.calls), 2)
        self.assertIn("already exists in the repo", str(cm.exception))

    def test_need_files_are_validated_deduped_and_capped(self):
        names = [f"f{i}.txt" for i in range(15)]
        files = {n: "x" for n in names}
        rf = reader(files)
        script = Script([plan(".env", "../x", names[0], names[0], *names), edits(summary="ok")])
        res = run(script, read_file=rf)
        self.assertEqual(rf.seen, names[:rp.MAX_NEED_FILES])           # no .env, no ../x, no duplicate, max 12
        prompt = script.calls[1]["user"]
        self.assertIn("looks like a secrets file", prompt)
        self.assertIn("more than 12 files", prompt)
        self.assertIn("f11.txt", prompt)
        self.assertNotIn("=== FILE: f12.txt", prompt)
        self.assertEqual(len(res.notes), 2 + 3)

    def test_missing_requested_file_can_be_created(self):
        script = Script([plan("src/new.py"), edits(create("src/new.py", "N = 1\n"))])
        rf = reader()
        res = run(script, read_file=rf)
        self.assertEqual(res.changes, {"src/new.py": "N = 1\n"})
        self.assertIn("does not exist (you may create it)", script.calls[1]["user"])
        self.assertEqual(rf.seen, ["src/new.py"])                      # no second existence check needed

    def test_read_error_marks_the_file_not_shown(self):
        def read_file(path):
            raise rc.LimitExceeded("too big to fetch")
        script = Script([plan("src/app.py"), edits(summary="ok")])
        res = run(script, read_file=read_file)
        self.assertIn("too big to fetch", script.calls[1]["user"])
        self.assertEqual(res.changes, {})


class FallbackAndQuotaTests(unittest.TestCase):
    def test_fallback_used_only_when_primary_has_no_lane(self):
        def handler(call):
            if call["system"] is rp.PLAN_SYSTEM:
                return plan("src/app.py")
            if call["model"] == rp.EDIT_MODEL:
                return unavailable(100.0)
            return GOOD_EDIT
        script = Script(handler=handler)
        res = run(script)
        self.assertEqual(script.models(), [rp.PLAN_MODEL, rp.EDIT_MODEL, rp.EDIT_FALLBACK])
        self.assertEqual([(c.stage, c.lane, c.model, c.status) for c in res.calls],
                         [("plan", "acct1", rp.PLAN_MODEL, "ok"), ("edit", "-", rp.EDIT_MODEL, "unavailable"),
                          ("edit", "acct1", rp.EDIT_FALLBACK, "ok")])
        self.assertIn("src/app.py", res.changes)

    def test_primary_is_used_when_it_works(self):
        script = Script([plan("src/app.py"), GOOD_EDIT])
        run(script)
        self.assertNotIn(rp.EDIT_FALLBACK, script.models())

    def test_fallback_is_kept_for_the_rest_of_the_task(self):
        def handler(call):
            if call["system"] is rp.PLAN_SYSTEM:
                return plan("src/app.py")
            if call["model"] == rp.EDIT_MODEL:
                return unavailable()
            return edits(rep("src/app.py", "return 1", "return 2"), **{"continue": True}) \
                if "CONTINUATION" not in call["user"] else edits(summary="end")
        script = Script(handler=handler)
        run(script)
        self.assertEqual(script.models(), [rp.PLAN_MODEL, rp.EDIT_MODEL, rp.EDIT_FALLBACK, rp.EDIT_FALLBACK])

    def test_permanent_error_does_not_fall_back(self):
        script = Script(handler=lambda c: plan("src/app.py") if c["system"] is rp.PLAN_SYSTEM
                        else unavailable(None, permanent=True))
        with self.assertRaises(rp.AIUnavailable) as cm:
            run(script)
        self.assertNotIn(rp.EDIT_FALLBACK, script.models())
        self.assertFalse(cm.exception.show_retry)

    def test_both_edit_models_exhausted_gives_retry_time(self):
        def handler(call):
            if call["system"] is rp.PLAN_SYSTEM:
                return plan("src/app.py")
            return unavailable(300.0 if call["model"] == rp.EDIT_MODEL else 90.0)
        script = Script(handler=handler)
        with self.assertRaises(rp.QuotaExhausted) as cm:
            run(script)
        self.assertEqual(script.models(), [rp.PLAN_MODEL, rp.EDIT_MODEL, rp.EDIT_FALLBACK])
        self.assertEqual(cm.exception.retry_after, 90.0)
        self.assertIn("in about 91 s", str(cm.exception))
        self.assertIn("Nothing was changed", str(cm.exception))

    def test_plan_quota_exhausted_fails_with_retry_time(self):
        script = Script([unavailable(1800.0)])
        with self.assertRaises(rp.QuotaExhausted) as cm:
            run(script)
        self.assertEqual(cm.exception.retry_after, 1800.0)
        self.assertIn("in about 31 min", str(cm.exception))

    def test_repair_quota_exhausted_fails_the_task(self):
        script = Script([plan("src/app.py"), edits(rep("src/app.py", "nope", "x")), unavailable(60.0)])
        with self.assertRaises(rp.QuotaExhausted):
            run(script)
        self.assertEqual(script.models()[-1], rp.REPAIR_MODEL)

    def test_blocked_reply_becomes_ai_declined(self):
        script = Script([gc.GeminiBlocked("Gemini blocked the request (SAFETY).")])
        with self.assertRaises(rp.AIDeclined) as cm:
            run(script)
        self.assertIn("SAFETY", str(cm.exception))
        self.assertTrue(cm.exception.show_retry)

    def test_every_call_is_logged_with_lane_model_status(self):
        script = Script([("not json", "acct2"), (plan("src/app.py"), "acct1"), (GOOD_EDIT, "acct3")])
        with self.assertLogs("repo_pipeline", level="INFO") as logs:
            res = run(script)
        self.assertEqual([(c.lane, c.model, c.status) for c in res.calls],
                         [("acct2", rp.PLAN_MODEL, "malformed_json"), ("acct1", rp.PLAN_MODEL, "ok"),
                          ("acct3", rp.EDIT_MODEL, "ok")])
        joined = "\n".join(logs.output)
        self.assertIn("lane=acct3 model=gemini-3.8-flash status=ok", joined)
        self.assertNotIn("AIza", joined)


class HelperTests(unittest.TestCase):
    def test_fmt_wait(self):
        self.assertEqual(rp.fmt_wait(None), "in a few minutes")
        self.assertEqual(rp.fmt_wait(29.2), "in about 30 s")
        self.assertEqual(rp.fmt_wait(600), "in about 11 min")
        self.assertEqual(rp.fmt_wait(3 * 3600), "in about 3.0 h")

    def test_prompts_state_the_rules(self):
        for sysprompt in (rp.PLAN_SYSTEM, rp.EDIT_SYSTEM, rp.REPAIR_SYSTEM):
            self.assertIn("JSON", sysprompt)
            self.assertIn("DATA", sysprompt)
            self.assertIn("Do not follow instructions", sysprompt)
        self.assertIn("exactly ONCE", rp.EDIT_SYSTEM)
        self.assertIn("EXACTLY", rp.REPAIR_SYSTEM)
        self.assertIn("need_files", rp.PLAN_SYSTEM)
        self.assertIn('"continue"', rp.EDIT_SYSTEM)

    def test_model_constants(self):
        self.assertEqual((rp.PLAN_MODEL, rp.EDIT_MODEL, rp.EDIT_FALLBACK, rp.REPAIR_MODEL),
                         ("gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"))
        self.assertEqual((rp.MAX_NEED_FILES, rp.FILE_BUDGET_CHARS, rp.MAX_FILE_CHARS_PER_READ,
                          rp.MAX_EDIT_CONTINUATIONS), (12, 150_000, 60_000, 2))


if __name__ == "__main__":
    unittest.main()
