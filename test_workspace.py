# test_workspace.py -- offline tests of workspace.py (edit parsing, deterministic applier, verification).
# Run (from the bot folder): python -m unittest test_workspace -v
import json
import sys
import types
import unittest
from unittest.mock import patch

if "botkey" not in sys.modules:
    bk = types.ModuleType("botkey")
    bk.bot_key = "DISCORDSECRET.aaaaaa.bbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    bk.GITHUB_TOKEN = "github_pat_" + "A" * 50
    bk.REPO_OWNER_IDS = [42]
    bk.REPO_ALLOWED_GUILD_IDS = [7]
    bk.SANDBOX_REPO = "me/sandbox"
    bk.REPO_ALLOWLIST = {"me/app": ["main"]}
    sys.modules["botkey"] = bk

import repoconfig as rc  # noqa: E402
import workspace as w  # noqa: E402


def reply(*edits, **extra):
    return json.dumps({"summary": " Did it. ", "edits": list(edits), **extra})


def rep(path, search, replace):
    return {"path": path, "op": "replace", "search": search, "replace": replace}


def E(path, op, search=None, replace=None):
    return w.Edit(path, op, search, replace)


class ParseTests(unittest.TestCase):
    def test_valid_reply(self):
        r = w.parse_edit_reply(reply(rep("a.py", "x", "y"), {"path": "b.py", "op": "create", "replace": "Z\n"},
                                     {"path": "c.py", "op": "delete"}, **{"continue": True}))
        self.assertEqual(r.summary, "Did it.")
        self.assertTrue(r.cont)
        self.assertEqual([(e.path, e.op) for e in r.edits], [("a.py", "replace"), ("b.py", "create"), ("c.py", "delete")])
        self.assertEqual((r.edits[0].search, r.edits[0].replace), ("x", "y"))

    def test_defaults_and_fenced_json(self):
        r = w.parse_edit_reply("```json\n" + reply() + "\n```")
        self.assertEqual((r.edits, r.cont), ([], False))
        self.assertEqual(w.parse_edit_reply('{"edits": []}').summary, "")

    def test_paths_are_normalized_and_checked(self):
        self.assertEqual(w.parse_edit_reply(reply(rep("./src//a.py".replace("//", "/"), "x", "y"))).edits[0].path, "src/a.py")
        for bad in ("../evil.py", "/abs.py", ".env", "a/.git/config", ".github/workflows/ci.yml", "logo.png"):
            with self.assertRaises(w.EditParseError, msg=bad) as cm:
                w.parse_edit_reply(reply({"path": bad, "op": "create", "replace": "x"}))
            self.assertIn("edits[0].path", str(cm.exception))

    def test_malformed_replies(self):
        cases = ["", "nope", "[]", '{"edits": "x"}', '{"summary": 1, "edits": []}', '{"edits": [], "continue": "yes"}',
                 reply("str"), reply({"path": "a", "op": "rename"}),
                 reply({"path": "a.py", "op": "replace", "search": "", "replace": "x"}),
                 reply({"path": "a.py", "op": "replace", "search": "x"}),
                 reply({"path": "a.py", "op": "create"}),
                 reply(*[{"path": "a.py", "op": "delete"}] * (w.MAX_EDITS_PER_REPLY + 1))]
        for bad in cases:
            with self.assertRaises(w.EditParseError, msg=bad[:40]):
                w.parse_edit_reply(bad)
        self.assertTrue(issubclass(w.EditParseError, rc.ValidationError))

    def test_crlf_in_model_text_becomes_lf(self):
        e = w.parse_edit_reply(reply(rep("a.py", "a\r\nb", "c\r\nd"))).edits[0]
        self.assertEqual((e.search, e.replace), ("a\nb", "c\nd"))


class ApplyTests(unittest.TestCase):
    def test_replace_create_delete_in_order(self):
        ws = w.Workspace({"a.py": "x = 1\ny = 2\n", "old.py": "1\n"})
        res = w.apply_edits(ws, [E("a.py", "replace", "x = 1", "x = 5"), E("a.py", "replace", "x = 5", "x = 6"),
                                 E("new.py", "create", None, "N\n"), E("old.py", "delete")])
        self.assertEqual((len(res.applied), res.failed), (4, []))
        self.assertEqual(ws.changes(), {"a.py": "x = 6\ny = 2\n", "new.py": "N\n", "old.py": None})
        self.assertEqual(ws.touched(), {"a.py", "new.py", "old.py"})
        self.assertFalse(ws.exists("old.py"))

    def test_failures_are_reported_and_change_nothing(self):
        ws = w.Workspace({"a.py": "x\nx\n"})
        res = w.apply_edits(ws, [E("a.py", "replace", "zzz", "q"), E("a.py", "replace", "x", "q"),
                                 E("ghost.py", "replace", "a", "b"), E("ghost.py", "delete"),
                                 E("a.py", "create", None, "over")])
        reasons = [why for _, why in res.failed]
        self.assertEqual(len(res.applied), 0)
        self.assertIn("not found", reasons[0])
        self.assertIn("2 times", reasons[1])
        self.assertIn("does not exist", reasons[2])
        self.assertIn("does not exist", reasons[3])
        self.assertIn("already exists", reasons[4])
        self.assertEqual((ws.changes(), ws.get("a.py")), ({}, "x\nx\n"))

    def test_good_edits_survive_a_failed_one(self):
        ws = w.Workspace({"a.py": "a = 1\n", "b.py": "b = 1\n"})
        res = w.apply_edits(ws, [E("a.py", "replace", "a = 1", "a = 2"), E("b.py", "replace", "nope", "x")])
        self.assertEqual((len(res.applied), len(res.failed)), (1, 1))
        self.assertEqual(ws.changes(), {"a.py": "a = 2\n"})

    def test_delete_then_create_replaces_a_file(self):
        ws = w.Workspace({"a.py": "old\n"})
        res = w.apply_edits(ws, [E("a.py", "delete"), E("a.py", "create", None, "new\n")])
        self.assertEqual(res.failed, [])
        self.assertEqual(ws.changes(), {"a.py": "new\n"})

    def test_unchanged_and_create_then_delete_leave_no_change(self):
        ws = w.Workspace({"a.py": "same\n"})
        w.apply_edits(ws, [E("a.py", "delete"), E("a.py", "create", None, "same\n"),
                           E("tmp.py", "create", None, "t\n"), E("tmp.py", "delete")])
        self.assertEqual(ws.changes(), {})

    def test_replacement_text_is_literal(self):
        ws = w.Workspace({"a.py": "p = 1\n"})
        w.apply_edits(ws, [E("a.py", "replace", "p = 1", r"p = '\1' % 2")])
        self.assertEqual(ws.get("a.py"), "p = '\\1' % 2\n")

    def test_crlf_files_are_matched_as_lf_and_restored(self):
        ws = w.Workspace({"a.py": "x = 1\r\ny = 2\r\n", "lf.py": "k = 1\n"})
        w.apply_edits(ws, [E("a.py", "replace", "x = 1\ny", "x = 9\ny"), E("lf.py", "replace", "1", "2"),
                           E("new.py", "create", None, "n\n")])
        self.assertEqual(ws.normalized, {"a.py"})
        self.assertEqual(ws.get("a.py"), "x = 9\ny = 2\n")
        self.assertEqual(ws.changes(restore_eol=True),
                         {"a.py": "x = 9\r\ny = 2\r\n", "lf.py": "k = 2\n", "new.py": "n\n"})
        self.assertEqual(ws.changes()["a.py"], "x = 9\ny = 2\n")


class LimitTests(unittest.TestCase):
    def test_file_size_limit(self):
        with patch.object(rc, "MAX_FILE_CHARS", 10):
            ws = w.Workspace({"a.txt": "12345"})
            res = w.apply_edits(ws, [E("a.txt", "replace", "5", "5" * 20), E("b.txt", "create", None, "x" * 11)])
        self.assertEqual(len(res.failed), 2)
        self.assertTrue(all(why.startswith(w.LIMIT_PREFIX) for _, why in res.failed))
        self.assertEqual(ws.changes(), {})

    def test_file_count_limit(self):
        with patch.object(rc, "MAX_FILES_PER_TASK", 2):
            ws = w.Workspace()
            res = w.apply_edits(ws, [E(f"f{i}.txt", "create", None, "x") for i in range(3)])
            again = w.apply_edits(ws, [E("f0.txt", "replace", "x", "y")])      # an already touched file is fine
        self.assertEqual((len(res.applied), len(res.failed)), (2, 1))
        self.assertTrue(res.failed[0][1].startswith(w.LIMIT_PREFIX))
        self.assertEqual(again.failed, [])

    def test_total_size_limit(self):
        with patch.object(rc, "MAX_TOTAL_CHARS", 15):
            ws = w.Workspace()
            res = w.apply_edits(ws, [E("a.txt", "create", None, "a" * 10), E("b.txt", "create", None, "b" * 10),
                                     E("a.txt", "replace", "a" * 10, "a" * 12)])      # growing a.txt by 2 still fits
        self.assertEqual([e.op for e in res.applied], ["create", "replace"])
        self.assertEqual([e.path for e, _ in res.failed], ["b.txt"])
        self.assertTrue(res.failed[0][1].startswith(w.LIMIT_PREFIX))


class VerifyTests(unittest.TestCase):
    def test_python_and_json(self):
        ws = w.Workspace({"ok.py": "x = 1\n", "bad.py": "def f(:\n", "ok.json": '{"a": 1}', "bad.json": '{"a": 1,}',
                          "notes.md": "(((", "gone.py": "x"})
        w.apply_edits(ws, [E("gone.py", "delete")])
        out = dict(w.verify(ws, ["ok.py", "bad.py", "ok.json", "bad.json", "notes.md", "gone.py", "missing.py"]))
        self.assertEqual(set(out), {"bad.py", "bad.json"})
        self.assertIn("SyntaxError", out["bad.py"])
        self.assertIn("JSON error", out["bad.json"])


if __name__ == "__main__":
    unittest.main()
