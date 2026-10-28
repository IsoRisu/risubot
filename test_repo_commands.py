# test_repo_commands.py -- offline tests for the Discord side of the repo feature (repocmds.py, repotarget.py).
# Run (from the bot folder): python -m unittest test_repo_commands -v
#
# Gemini never gets called for real: unit tests inject a scripted `call` hook (the same hook repo_pipeline uses),
# and the EndToEndTests go one level lower and script the HTTP transport under gemini_client.call_gemini, so lane
# rotation, quota counters, the pipeline and the Discord/GitHub side all run for real. GitHub is a fake client
# (signatures checked against the real one in ContractTests) or the real client over a fake wire.
import asyncio
import functools
import inspect
import json
import sys
import time
import types
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

bk = sys.modules.get("botkey") or types.ModuleType("botkey")
for _k, _v in dict(bot_key="DISCORDSECRET.aaaaaa.bbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                   GITHUB_TOKEN="github_pat_" + "A" * 50,
                   REPO_OWNER_IDS=[42], REPO_ALLOWED_GUILD_IDS=[7],
                   SANDBOX_REPO="me/sandbox", REPO_ALLOWLIST={"me/app": ["main"]}).items():
    setattr(bk, _k, _v)
sys.modules["botkey"] = bk

try:
    import discord  # noqa: F401
except ImportError:                     # minimal stand-in so repotarget imports without discord.py
    class _Command:
        def __init__(self, callback):
            self.callback = callback

    class _Group:
        def __init__(self, name, description="", guild_only=False):
            self.name, self.commands = name, {}

        def command(self, name, description=""):
            def deco(fn):
                self.commands[name] = _Command(fn)
                return fn
            return deco

        def get_command(self, name):
            return self.commands.get(name)

    class _Range:
        def __class_getitem__(cls, item):
            return str

    discord = types.ModuleType("discord")
    discord.AllowedMentions = SimpleNamespace(none=lambda: None)
    discord.Interaction = object
    discord.Object = lambda id: SimpleNamespace(id=id)
    discord.HTTPException = type("HTTPException", (Exception,), {"status": 0})
    app_commands = types.ModuleType("discord.app_commands")
    app_commands.Group, app_commands.Range = _Group, _Range
    app_commands.describe = lambda **kw: (lambda fn: fn)
    discord.app_commands = app_commands
    sys.modules["discord"], sys.modules["discord.app_commands"] = discord, app_commands

import gemini_client as gc  # noqa: E402
import gemini_lanes as gl  # noqa: E402
import githubclient as gh  # noqa: E402
import repo_pipeline as rp  # noqa: E402
import repocmds  # noqa: E402
import repoconfig as rc  # noqa: E402
import repotarget  # noqa: E402
import repotask  # noqa: E402

SPEC = rc.resolve_target("me/app")
PR_URL = "https://github.com/me/app/pull/9"
APP = "def f():\n    return 1\n"
README = "# App\nA small app.\n"
NOON = datetime(2026, 10, 28, 12, 0, tzinfo=ZoneInfo(gl.QUOTA_TZ)).timestamp()


# ---------------------------------------------------------------------------
# helpers and fakes
# ---------------------------------------------------------------------------
def run(coro):
    return asyncio.run(coro)


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


GOOD_EDIT = edits(rep("src/app.py", "return 1", "return 2"), create("src/new.py", "X = 1\n"),
                  summary="Changed f and added new.py.")


class Script:
    """A fake `call` hook for repo_pipeline: replays `items` (reply text, or an Exception to raise) and records
    every call as {model, system, user, kw}."""

    def __init__(self, items):
        self.items, self.calls = iter(items), []

    def __call__(self, store, lanes, model, system_prompt, user_text, **kw):
        self.calls.append(dict(model=model, system=system_prompt, user=user_text, kw=kw))
        item = next(self.items)
        if isinstance(item, Exception):
            raise item
        return gc.GeminiReply(item, "acct1", model, "STOP")

    def models(self):
        return [c["model"] for c in self.calls]


def backend_for(script):
    return repocmds.Backend(object(), [], script)


class FakeClient:
    """Stands in for GitHubClient. Reads come from `files`; writes are recorded. The signatures of
    everything repotarget/repocmds call must match the real client (see ContractTests)."""

    def __init__(self, files=None, latest=None, pr_error=None, modes=None):
        self.files = files if files is not None else {"README.md": README, "src/app.py": APP}
        self.modes = modes or {}
        self.latest, self.pr_error = latest, pr_error
        self.view_calls, self.latest_calls, self.commits, self.prs, self.pushed = [], [], [], [], None

    def view(self, repo, ref):
        self.view_calls.append((repo, ref))
        return gh.RepoView(self, repo, ref)

    def list_tree(self, repo, ref):
        return gh.TreeListing("sha", [gh.TreeEntry(p, len(t), self.modes.get(p, "100644"), "s")
                                      for p, t in self.files.items()])

    def read_file(self, repo, path, ref):
        if path not in self.files:
            raise gh.GitHubNotFound("not found", status=404)
        return gh.FileContent(path, self.files[path], len(self.files[path]), "s")

    def latest_bot_branch(self, repo):
        self.latest_calls.append(repo)
        return self.latest

    def commit_to_branch(self, repo, branch, base, changes, message, modes=None):
        self.commits.append((repo, dict(branch=branch, base=base, changes=changes, message=message, modes=modes)))
        return gh.CommitResult("c1", "https://github.com/me/app/commit/c1", branch)

    def open_pr(self, repo, head, base, title, body):
        if self.pr_error:
            raise self.pr_error
        self.prs.append((repo, dict(head=head, base=base, title=title, body=body)))
        return SimpleNamespace(url=PR_URL)

    def sandbox_force_push(self, files, message, branch=None):
        self.pushed = (dict(files), message, branch)
        return gh.CommitResult("c1", "https://github.com/me/sandbox/commit/c1", branch or "main")


class FakeBot:
    def __init__(self):
        self.tree = SimpleNamespace(add_command=lambda *a, **kw: None)

    async def setup_hook(self):
        pass


class FakeInteraction:
    """Records what the command does: ("defer", kw) / ("response", text, kw) / ("followup", text, kw)."""

    def __init__(self, user_id=42, guild_id=7):
        self.user, self.guild_id, self.channel_id, self.channel = SimpleNamespace(id=user_id), guild_id, 1, None
        self.ev, self._done = [], False
        self.response = SimpleNamespace(defer=self._defer, is_done=lambda: self._done, send_message=self._respond)
        self.followup = SimpleNamespace(send=self._followup)

    async def _defer(self, **kw):
        self._done = True
        self.ev.append(("defer", kw))

    async def _respond(self, text, **kw):
        self._done = True
        self.ev.append(("response", text, kw))

    async def _followup(self, text, **kw):
        self.ev.append(("followup", text, kw))

    def kinds(self):
        return [e[0] for e in self.ev]


def job(script, client, prompt="add src/new.py", **kw):
    kw.setdefault("read_only", False)
    return run(repotarget.target_job(SPEC, prompt, client=client, backend=backend_for(script), **kw))


def unavailable(retry_after=11.0, permanent=False):
    return gc.GeminiUnavailable("out of quota", retry_after=retry_after, permanent=permanent)


# ---------------------------------------------------------------------------
# README.md / AGENTS.md notes
# ---------------------------------------------------------------------------
class NotesTests(unittest.TestCase):
    @staticmethod
    def reader(files):
        return lambda path: files.get(path)

    def test_budget_is_6000_chars(self):
        self.assertEqual(repocmds.MAX_NOTES_CHARS, 6_000)

    def test_nothing_or_blank_gives_empty_string(self):
        self.assertEqual(repocmds.build_notes(self.reader({})), "")
        self.assertEqual(repocmds.build_notes(self.reader({"README.md": " \n", "AGENTS.md": ""})), "")

    def test_both_files_are_labelled_in_order(self):
        text = repocmds.build_notes(self.reader({"README.md": "readme body", "AGENTS.md": "agents body"}))
        self.assertEqual(text, "=== README.md ===\nreadme body\n\n=== AGENTS.md ===\nagents body")

    def test_only_agents_md(self):
        self.assertEqual(repocmds.build_notes(self.reader({"AGENTS.md": "rules"})), "=== AGENTS.md ===\nrules")

    def test_truncated_to_the_budget_with_a_visible_marker(self):
        files = {"README.md": "r" * 20_000, "AGENTS.md": "a" * 20_000}
        text = repocmds.build_notes(self.reader(files))
        self.assertLessEqual(len(text), repocmds.MAX_NOTES_CHARS)
        self.assertGreater(len(text), repocmds.MAX_NOTES_CHARS - 5)
        self.assertEqual(text.count("[... truncated]"), 2)
        self.assertIn("=== AGENTS.md ===", text)          # the second file is not starved by the first

    def test_a_short_file_keeps_all_of_its_text(self):
        files = {"README.md": "r" * 20_000, "AGENTS.md": "short rules"}
        text = repocmds.build_notes(self.reader(files))
        self.assertLessEqual(len(text), repocmds.MAX_NOTES_CHARS)
        self.assertIn("=== AGENTS.md ===\nshort rules", text)
        self.assertEqual(text.count("[... truncated]"), 1)

    def test_unreadable_file_is_skipped(self):
        def read(path):
            if path == "README.md":
                raise rc.LimitExceeded("too large")
            return "rules"
        self.assertEqual(repocmds.build_notes(read), "=== AGENTS.md ===\nrules")

    def test_tiny_limit_never_overflows(self):
        text = repocmds.build_notes(self.reader({"README.md": "x" * 500, "AGENTS.md": "y" * 500}), limit=60)
        self.assertLessEqual(len(text), 60)


# ---------------------------------------------------------------------------
# the read_file adapter for the pipeline
# ---------------------------------------------------------------------------
class ReaderTests(unittest.TestCase):
    def test_text_missing_binary_and_oversize(self):
        client = FakeClient({"a.txt": "hello", "b.bin": "x"})
        view = client.view("me/app", "main")
        read = repocmds.make_reader(view)
        self.assertEqual(read("a.txt"), "hello")
        self.assertIsNone(read("nope.txt"))
        client.read_file = lambda repo, path, ref: gh.FileContent(path, "", 4, "s", is_binary=True)
        with self.assertRaises(rc.ValidationError):
            repocmds.make_reader(client.view("me/app", "main"))("b.bin")

        def too_big(repo, path, ref):
            raise rc.LimitExceeded("too large")
        client.read_file = too_big
        with self.assertRaises(rc.LimitExceeded):
            repocmds.make_reader(client.view("me/app", "main"))("big.txt")

    def test_other_github_errors_are_not_swallowed(self):
        client = FakeClient()

        def boom(repo, path, ref):
            raise gh.GitHubRateLimit("slow down", status=403)
        client.read_file = boom
        with self.assertRaises(gh.GitHubRateLimit):
            repocmds.make_reader(client.view("me/app", "main"))("README.md")


# ---------------------------------------------------------------------------
# the process-wide Gemini backend
# ---------------------------------------------------------------------------
class BackendTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(repocmds, "_backend", None)
        p.start()
        self.addCleanup(p.stop)

    def test_built_once_from_botkey_config_with_the_persistent_quota_file(self):
        lanes = [gl.Lane("acct1", ["k1"]), gl.Lane("acct2", ["k2"])]
        with patch.object(gl, "load_lanes", return_value=lanes), \
                patch.object(gl, "load_limits", return_value=(dict(gl.DEFAULT_RPD), dict(gl.DEFAULT_RPM))), \
                patch.object(repocmds, "QUOTA_FILE", None):
            first = repocmds.default_backend()
            self.assertIs(repocmds.default_backend(), first)
        self.assertEqual([lane.id for lane in first.lanes], ["acct1", "acct2"])
        self.assertIs(first.store.lanes[0], lanes[0])
        self.assertEqual(first.store.rpd["gemini-3.8-flash"], 20)
        self.assertIsNone(first.call)
        self.assertTrue(repocmds.QUOTA_FILE.endswith("gemini_quota.json"))

    def test_no_keys_is_a_config_error_without_retry_hint(self):
        with patch.object(gl, "load_lanes", return_value=[]), patch.object(gl, "load_limits", return_value=({}, {})):
            with self.assertRaises(rc.ConfigError) as cm:
                repocmds.default_backend()
        self.assertIn("GEMINI_LANES", str(cm.exception))
        self.assertNotIn("Retry with", repotask.format_error(cm.exception, "!code x"))

    def test_bad_config_is_a_config_error(self):
        with patch.object(gl, "load_lanes", side_effect=ValueError("GEMINI_LANES[0] is empty.")):
            with self.assertRaises(rc.ConfigError) as cm:
                repocmds.default_backend()
        self.assertIn("GEMINI_LANES[0] is empty", str(cm.exception))


# ---------------------------------------------------------------------------
# sandbox flow (!code)
# ---------------------------------------------------------------------------
class SandboxTests(unittest.TestCase):
    def test_parse_args(self):
        self.assertEqual(repocmds.parse_code_args("--branch demo make a game"), ("demo", "make a game"))
        self.assertEqual(repocmds.parse_code_args("--branch=demo x"), ("demo", "x"))
        self.assertEqual(repocmds.parse_code_args("make --branch x"), (None, "make --branch x"))
        with self.assertRaises(rc.ValidationError):
            repocmds.parse_code_args("--branch ../x go")

    def test_job_pushes_files(self):
        script = Script([plan(), edits(create("game/main.py", "print('hi')\n"), summary="Wrote the game.")])
        client = FakeClient()
        msg = run(repocmds.sandbox_job("make a game", "demo", client=client, backend=backend_for(script)))
        self.assertEqual(client.pushed, ({"game/main.py": "print('hi')\n"}, "Sandbox output (1 file(s))", "demo"))
        self.assertIn("commit/c1", msg)
        self.assertIn("Wrote the game.", msg)
        self.assertEqual(client.commits, [])                             # the sandbox never uses the PR path
        plan_call = script.calls[0]
        self.assertEqual(script.models(), [rp.PLAN_MODEL, rp.EDIT_MODEL])
        self.assertIn("make a game", plan_call["user"])
        self.assertIn(repocmds.SANDBOX_TREE, plan_call["user"])

    def test_empty_result_does_not_wipe_the_sandbox(self):
        client = FakeClient()
        script = Script([plan(), edits(summary="Hello!")])
        self.assertEqual(run(repocmds.sandbox_job("hi", None, client=client, backend=backend_for(script))), "Hello!")
        self.assertIsNone(client.pushed)

    def test_the_sandbox_reader_finds_no_files(self):
        """Edits on files that were never shown fail like anywhere else; nothing is pushed."""
        script = Script([plan(), edits(rep("a.py", "x", "y")), J(summary="no", edits=[])])
        client = FakeClient()
        with self.assertRaises(rp.EditsFailed):
            run(repocmds.sandbox_job("x", None, client=client, backend=backend_for(script)))
        self.assertIsNone(client.pushed)

    def test_outage_error_has_retry_command_and_wait(self):
        sent = []

        async def send(m):
            sent.append(m)

        script = Script([unavailable(11.0)])

        async def go():
            task = repotask.TaskContext("sandbox", 42, 7, 1, "!code make x")
            return await repotask.run_task(
                task, send, lambda t: repocmds.sandbox_job("x", None, client=FakeClient(), backend=backend_for(script)),
                lock=repotask.TaskLock())
        self.assertFalse(run(go()))
        self.assertIn("Retry with: `!code make x`", sent[0])
        self.assertIn("12 s", sent[0])
        self.assertNotIn("Remote AI", sent[0])


# ---------------------------------------------------------------------------
# target repos: names, commit + PR, iterate, slash command
# ---------------------------------------------------------------------------
class NameTests(unittest.TestCase):
    def test_bot_branch_name_forces_prefix(self):
        self.assertEqual(repotarget.bot_branch_name("demo", "main"), "ai/demo")
        self.assertEqual(repotarget.bot_branch_name("ai/demo", "main"), "ai/demo")
        self.assertEqual(repotarget.bot_branch_name("main", "main"), "ai/main")      # can never hit the base
        with self.assertRaises(rc.ValidationError):
            repotarget.bot_branch_name("../x", "main")

    def test_new_branch_name(self):
        self.assertRegex(repotarget.new_branch_name("Fix the Login bug!! in src/app.py now", now=0),
                         r"^ai/fix-the-login-bug-\d{4}-\d{6}$")
        self.assertRegex(repotarget.new_branch_name("???"), r"^ai/task-")

    def test_default_title_body_and_retry_text(self):
        self.assertEqual(repotarget.default_title("\n\n  add   a   thing \nmore"), "add a thing")
        self.assertEqual(len(repotarget.default_title("x" * 300)), repotarget.DEFAULT_TITLE_CHARS)
        self.assertEqual(repotarget.default_title("   "), "AI changes")
        body = repotarget.default_body("ping @everyone", "done by @bob")
        self.assertNotIn("@everyone", body)
        self.assertNotIn("@bob", body)
        self.assertIn("AI summary", body)
        self.assertEqual(repotarget.retry_text("a/b", "do it", "dev", "ai/x", True, "T"),
                         "/repo task repo:a/b prompt:do it base:dev branch:ai/x iter:True title:T")

    def test_existing_modes_only_for_changed_existing_files(self):
        listing = gh.TreeListing("s", [gh.TreeEntry("run.sh", 1, "100755", "s"), gh.TreeEntry("a.py", 1, "100644", "s"),
                                       gh.TreeEntry("link", 1, "120000", "s"), gh.TreeEntry("other.sh", 1, "100755", "s")])
        changes = {"run.sh": "x", "a.py": "y", "link": "z", "new.py": "n", "other.sh": None}
        self.assertEqual(repotarget.existing_modes(listing, changes), {"run.sh": "100755", "a.py": "100644"})


class WriteJobTests(unittest.TestCase):
    def test_commit_and_pr(self):
        client = FakeClient(modes={"src/app.py": "100755"})
        script = Script([plan("src/app.py"), GOOD_EDIT])
        text = job(script, client)
        (repo, kw), = client.commits
        self.assertEqual(repo, "me/app")
        self.assertEqual((kw["base"], kw["changes"]),
                         ("main", {"src/app.py": "def f():\n    return 2\n", "src/new.py": "X = 1\n"}))
        self.assertEqual(kw["modes"], {"src/app.py": "100755"})           # exec bit kept; new files get no entry
        self.assertTrue(kw["branch"].startswith("ai/"))
        (_, pr), = client.prs
        self.assertEqual((pr["head"], pr["base"], pr["title"]), (kw["branch"], "main", "add src/new.py"))
        self.assertIn("Changed f and added new.py.", pr["body"])
        for part in (PR_URL, "`src/app.py`", "`src/new.py`", "Changed f and added new.py.", "2 AI call(s)"):
            self.assertIn(part, text)
        self.assertNotIn("read-only", text)
        self.assertEqual(client.view_calls, [("me/app", "main")])
        plan_call, edit_call = script.calls
        for needle in ("add src/new.py", "me/app", "src/app.py", "# App", "You may change files"):
            self.assertIn(needle, plan_call["user"])                      # request header, tree, README notes
        self.assertNotIn("READ-ONLY", plan_call["user"])
        self.assertIn("return 1", edit_call["user"])                      # the file the plan asked for

    def test_no_changes_means_no_commit(self):
        client = FakeClient()
        text = job(Script([plan(), edits(summary="Nothing to change.")]), client)
        self.assertEqual((client.commits, client.prs), ([], []))
        self.assertIn("No changes", text)
        self.assertIn("Nothing to change.", text)
        self.assertNotIn("⚠️", text)

    def test_custom_branch_title_body(self):
        client = FakeClient()
        job(Script([plan("src/app.py"), GOOD_EDIT]), client, branch="ai/demo", title="My title", body="Hello @team")
        self.assertEqual(client.commits[0][1]["branch"], "ai/demo")
        self.assertTrue(client.commits[0][1]["message"].startswith("My title"))
        pr = client.prs[0][1]
        self.assertEqual(pr["title"], "My title")
        self.assertNotIn("@team", pr["body"])
        self.assertTrue(pr["body"].startswith("Hello"))

    def test_pr_failure_keeps_the_branch_in_the_message(self):
        client = FakeClient(pr_error=RuntimeError("HTTP 422"))
        with self.assertLogs("repotarget", level="ERROR"):
            text = job(Script([plan("src/app.py"), GOOD_EDIT]), client)
        self.assertEqual(len(client.commits), 1)
        for part in ("⚠️", client.commits[0][1]["branch"], "HTTP 422", "commit/c1"):
            self.assertIn(part, text)

    def test_commit_failure_propagates_and_opens_no_pr(self):
        client = FakeClient()

        def boom(repo, **kw):
            raise rc.ValidationError("branch exists")
        client.commit_to_branch = boom
        with self.assertRaises(rc.ValidationError):
            job(Script([plan("src/app.py"), GOOD_EDIT]), client)
        self.assertEqual(client.prs, [])

    def test_pipeline_failure_commits_nothing(self):
        client = FakeClient()
        with self.assertRaises(rp.AIOutputError):
            job(Script([plan(), "not json", "still not json"]), client)
        self.assertEqual((client.commits, client.prs), ([], []))

    def test_read_only_default_discards_edits(self):
        client = FakeClient()
        script = Script([plan("src/app.py"), GOOD_EDIT])
        text = job(script, client, read_only=True)
        self.assertEqual((client.commits, client.prs), ([], []))
        self.assertIn("read-only", text)
        self.assertIn("⚠️", text)
        self.assertIn("READ-ONLY", script.calls[0]["user"])

    def test_the_real_client_refuses_a_secret_in_an_ai_edit(self):
        calls = []
        client = make_wire_client(calls)
        leak = create("src/leak.py", "TOKEN = '" + "github_pat_" + "A" * 50 + "'\n")      # the bot's own GitHub token
        with self.assertRaises(rc.SecretLeakError):
            job(Script([plan(), edits(leak)]), client)
        self.assertEqual(calls, [])                                       # refused before any write request


class SecretScanTests(unittest.TestCase):
    def test_keys_nested_in_gemini_lanes_are_found(self):
        bk.GEMINI_LANES = [["lane-one-key-0001", "lane-one-key-0002"], ["lane-two-key-0001"]]
        rc.secret_values.cache_clear()
        try:
            self.assertIsNotNone(rc.find_secret("X = 'lane-two-key-0001'"))
            self.assertIsNotNone(rc.find_secret("lane-one-key-0002"))
            self.assertIsNone(rc.find_secret("nothing secret here"))
        finally:
            del bk.GEMINI_LANES
            rc.secret_values.cache_clear()


class IterTests(unittest.TestCase):
    def test_iter_uses_newest_bot_branch(self):
        client = FakeClient(latest="ai/old-1")
        script = Script([plan("src/app.py"), GOOD_EDIT])
        text = job(script, client, iterate=True)
        self.assertEqual(client.latest_calls, ["me/app"])
        self.assertEqual(client.view_calls, [("me/app", "ai/old-1")])
        kw = client.commits[0][1]
        self.assertEqual((kw["branch"], kw["base"]), ("ai/old-1", "main"))
        self.assertEqual(client.prs[0][1]["head"], "ai/old-1")
        self.assertIn("ai/old-1", script.calls[0]["user"])
        self.assertIn("(continued)", text)

    def test_iter_with_named_branch_skips_lookup(self):
        client = FakeClient(latest="ai/other")
        job(Script([plan("src/app.py"), GOOD_EDIT]), client, iterate=True, branch="ai/mine")
        self.assertEqual(client.latest_calls, [])
        self.assertEqual(client.view_calls, [("me/app", "ai/mine")])

    def test_iter_without_any_bot_branch_is_a_validation_error(self):
        client = FakeClient(latest=None)
        script = Script([])
        with self.assertRaises(rc.ValidationError):
            job(script, client, iterate=True)
        self.assertEqual((client.view_calls, script.calls), ([], []))


class SlashTests(unittest.TestCase):
    def invoke(self, script, client, **args):
        group = repotarget.register(FakeBot(), client_factory=lambda: client, backend=backend_for(script), sync=False)
        it = FakeInteraction()
        args = {"repo": "me/app", "prompt": "add src/new.py", "base": None, "branch": None, "iter": False,
                "title": None, "body": None, **args}
        run(group.get_command("task").callback(it, **args))
        return it

    def test_write_run_posts_pr_link(self):
        client = FakeClient()
        it = self.invoke(Script([plan("src/app.py"), GOOD_EDIT]), client, branch="demo", title="  Add new  ")
        self.assertEqual(it.kinds()[0], "defer")
        self.assertTrue(any(PR_URL in e[1] for e in it.ev if e[0] == "followup"))
        self.assertEqual(client.commits[0][1]["branch"], "ai/demo")
        self.assertEqual(client.prs[0][1]["title"], "Add new")

    def test_iter_flag_reaches_the_job_and_failure_has_retry(self):
        client = FakeClient(latest="ai/old-1")
        self.invoke(Script([plan("src/app.py"), GOOD_EDIT]), client, iter=True)
        self.assertEqual(client.view_calls, [("me/app", "ai/old-1")])
        it = self.invoke(Script([]), FakeClient(latest=None), iter=True)
        self.assertEqual(it.kinds()[0], "defer")
        self.assertIn("no bot branch", it.ev[1][1].lower())
        self.assertIn("iter:True", it.ev[1][1])

    def test_bad_input_is_refused_before_defer(self):
        too_long = {"title": "t" * (repotarget.MAX_TITLE_CHARS + 1)}, {"body": "b" * (repotarget.MAX_BODY_CHARS + 1)}
        for bad in ({"branch": "../x"}, *too_long):
            client, script = FakeClient(), Script([])
            it = self.invoke(script, client, **bad)
            self.assertEqual(it.kinds(), ["response"])
            self.assertTrue(it.ev[0][2]["ephemeral"])
            self.assertEqual((client.view_calls, client.commits, script.calls), ([], [], []))

    def test_quota_exhaustion_shows_the_wait_and_the_retry_command(self):
        it = self.invoke(Script([unavailable(30.0)]), FakeClient(), branch="demo")
        text = it.ev[1][1]
        self.assertIn("Retry with: `/repo task repo:me/app prompt:add src/new.py branch:demo`", text)
        self.assertIn("about 31 s", text)
        self.assertIn("Nothing was changed", text)

    def test_every_pipeline_error_is_formatted_by_format_error(self):
        errors = [rp.AIUnavailable("Gemini is down."), rp.AIDeclined("Gemini declined this request (x)."),
                  rp.AIOutputError("The AI returned an unusable reply twice."), rp.EditsFailed("Edits failed."),
                  rp.TaskTooLarge("Too large."), rp.QuotaExhausted("Out of quota. Try again in about 5 min.")]
        for err in errors:
            with self.subTest(err=type(err).__name__):
                client = FakeClient()
                it = self.invoke(Script([err]), client)       # raised from inside the call hook: propagates as is
                self.assertTrue(it.ev[1][1].startswith("❌ " + str(err)[:20]), it.ev[1][1])
                self.assertEqual(client.commits, [])
                self.assertIn("Retry with:", it.ev[1][1])


# ---------------------------------------------------------------------------
# end to end: fake GitHub + scripted Gemini HTTP transport (lanes, quota, pipeline, commit, PR all real)
# ---------------------------------------------------------------------------
class Transport:
    """Scripted transport under gemini_client.call_gemini: replies[i] is the text of the i-th model answer."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, headers, body, timeout):
        payload = json.loads(body)
        self.calls.append(SimpleNamespace(
            model=url.split("/models/")[1].split(":")[0], key=headers["x-goog-api-key"],
            user=payload["contents"][0]["parts"][0]["text"],
            system=payload["systemInstruction"]["parts"][0]["text"], config=payload["generationConfig"]))
        text = self.replies.pop(0)
        return 200, {}, json.dumps({"candidates": [{"content": {"parts": [{"text": text}]},
                                                    "finishReason": "STOP"}]}).encode()

    def models(self):
        return [c.model for c in self.calls]


def e2e_backend(transport, ids=("a", "b"), rpd=None, clock=time.time):
    lanes = [gl.Lane(i, [f"key-{i}-1"]) for i in ids]
    store = gl.QuotaStore(lanes, None, rpd=dict(gl.DEFAULT_RPD if rpd is None else rpd),
                          rpm={m: 100 for m in gl.DEFAULT_RPM}, clock=clock)
    call = functools.partial(gc.call_gemini, transport=transport, health=gc.LaneHealth(clock=time.monotonic),
                             sleep=lambda s: None)
    return repocmds.Backend(store, lanes, call)


class EndToEndTests(unittest.TestCase):
    def test_slash_task_from_request_to_pull_request(self):
        client = FakeClient({"README.md": README, "AGENTS.md": "Use 4 spaces.\n", "src/app.py": APP,
                             "src/util.py": "Y = 1\n"}, modes={"src/app.py": "100755"})
        transport = Transport([plan("src/app.py", text="make f return 2"), GOOD_EDIT])
        backend = e2e_backend(transport)
        group = repotarget.register(FakeBot(), client_factory=lambda: client, backend=backend, sync=False)
        it = FakeInteraction()
        run(group.get_command("task").callback(it, repo="me/app", prompt="make f return 2", base=None, branch="demo",
                                               iter=False, title=None, body=None))
        # Discord side
        self.assertEqual(it.kinds(), ["defer", "followup"])
        msg = it.ev[1][1]
        for part in (PR_URL, "branch `ai/demo`", "`src/app.py`", "`src/new.py`", "Changed f and added new.py.",
                     "2 AI call(s)"):
            self.assertIn(part, msg)
        # GitHub side: built from the TaskResult, nothing else touched git
        (repo, kw), = client.commits
        self.assertEqual((repo, kw["branch"], kw["base"]), ("me/app", "ai/demo", "main"))
        self.assertEqual(kw["changes"], {"src/app.py": "def f():\n    return 2\n", "src/new.py": "X = 1\n"})
        self.assertEqual(kw["modes"], {"src/app.py": "100755"})
        self.assertEqual(client.prs[0][1]["head"], "ai/demo")
        # Gemini side: plan -> edit, JSON mode, 4000-token edit limit, tree + notes + file content in the prompts
        self.assertEqual(transport.models(), [rp.PLAN_MODEL, rp.EDIT_MODEL])
        plan_call, edit_call = transport.calls
        self.assertEqual(plan_call.config["responseMimeType"], "application/json")
        self.assertEqual(edit_call.config["maxOutputTokens"], 4000)
        for needle in ("make f return 2", "src/util.py", "# App", "Use 4 spaces."):
            self.assertIn(needle, plan_call.user)
        self.assertIn("return 1", edit_call.user)
        self.assertNotIn("Y = 1", edit_call.user)                         # util.py was not requested
        # quota: two calls were counted, spread over the two lanes (most remaining RPD first)
        self.assertEqual([backend.store.remaining(i, rp.PLAN_MODEL) for i in ("a", "b")], [19, 19])
        self.assertEqual({c.key for c in transport.calls}, {"key-a-1", "key-b-1"})

    def test_failed_anchor_goes_through_one_repair_call_on_the_lite_model(self):
        client = FakeClient()
        transport = Transport([plan("src/app.py"),
                               edits(rep("src/app.py", "return 99", "return 2"), summary="Tried."),
                               edits(rep("src/app.py", "return 1", "return 2"), summary="Fixed the anchor.")])
        text = run(repotarget.target_job(SPEC, "make f return 2", client=client, backend=e2e_backend(transport),
                                         read_only=False))
        self.assertEqual(transport.models(), [rp.PLAN_MODEL, rp.EDIT_MODEL, rp.REPAIR_MODEL])
        self.assertEqual(client.commits[0][1]["changes"], {"src/app.py": "def f():\n    return 2\n"})
        self.assertIn(PR_URL, text)
        self.assertIn("3 AI call(s)", text)

    def test_exhausted_quota_fails_with_the_retry_time_and_commits_nothing(self):
        client = FakeClient()
        transport = Transport([])
        backend = e2e_backend(transport, rpd={m: 0 for m in gl.DEFAULT_RPD}, clock=lambda: NOON)
        sent = []

        async def send(m):
            sent.append(m)

        async def go():
            task = repotask.TaskContext("target", 42, 7, 1, "/repo task repo:me/app prompt:x")
            return await repotask.run_task(
                task, send, lambda t: repotarget.target_job(SPEC, "x", client=client, backend=backend, read_only=False),
                lock=repotask.TaskLock())
        self.assertFalse(run(go()))
        self.assertEqual(transport.calls, [])                             # no request was sent
        self.assertEqual((client.commits, client.prs), ([], []))
        self.assertIn("Try again in about 12.0 h", sent[0])               # Pacific midnight is 12 h after NOON
        self.assertIn("Retry with: `/repo task repo:me/app prompt:x`", sent[0])
        self.assertIn("Nothing was changed", sent[0])

    def test_sandbox_code_command_flow_pushes_one_commit(self):
        client = FakeClient()
        transport = Transport([plan(), edits(create("index.html", "<h1>hi</h1>\n"), create("app.js", "go();\n"),
                                             summary="A tiny page.")])
        msg = run(repocmds.sandbox_job("a tiny page", None, client=client, backend=e2e_backend(transport)))
        self.assertEqual(client.pushed, ({"index.html": "<h1>hi</h1>\n", "app.js": "go();\n"},
                                         "Sandbox output (2 file(s))", None))
        self.assertIn("Pushed 2 file(s) to the sandbox", msg)
        self.assertIn("A tiny page.", msg)
        self.assertEqual(transport.models(), [rp.PLAN_MODEL, rp.EDIT_MODEL])


# ---------------------------------------------------------------------------
# the Gemini-only cut: nothing of the old tool loop is left in the Discord modules
# ---------------------------------------------------------------------------
class RemovalTests(unittest.TestCase):
    # The needles are assembled from pieces so that a repo-wide grep for the removed names finds only real leftovers.
    GONE = ("llm" + "bridge", "tool" + "loop", "ai" + "tools", "llm" + "tools", "LOC" + "AL_", "Remote AI unavailable",
            "chat" + "_tools_local", "Tool" + "Session", "olla" + "ma")

    def test_no_references_to_removed_code(self):
        for mod in (repocmds, repotarget):
            src = inspect.getsource(mod)
            for needle in self.GONE:
                self.assertNotIn(needle, src, f"{mod.__name__} still mentions {needle}")

    def test_removed_modules_are_gone(self):
        import importlib
        for name in ("llm" + "bridge", "tool" + "loop", "ai" + "tools", "llm" + "tools"):
            with self.assertRaises(ImportError, msg=name):
                importlib.import_module(name)

    def test_llmclient_is_remote_only(self):
        import os
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "llmclient.py"), encoding="utf-8") as f:
            src = f.read()            # read as text: importing llmclient needs a real provider key
        self.assertNotIn('"local"', src)
        self.assertNotIn("def local" + "_provider", src)


# ---------------------------------------------------------------------------
# the fake client must not drift from the real one
# ---------------------------------------------------------------------------
def make_wire_client(calls):
    """The real GitHubClient write path (commit_to_branch, open_pr, latest_bot_branch) with only the HTTP layer
    replaced; reads are served from memory by the fake. `calls` collects (method, path) of every request."""
    class WireClient(gh.GitHubClient):
        view, list_tree, read_file = FakeClient.view, FakeClient.list_tree, FakeClient.read_file

        def __init__(self):
            super().__init__(token="t")
            self.files, self.view_calls, self.modes = {"README.md": README, "src/app.py": APP}, [], {}

        def _request(self, method, path, *, body=None, params=None, **kw):
            calls.append((method, path))
            if path.endswith("/git/ref/heads/main"):
                return {"object": {"sha": "basesha"}}
            if "/git/ref/heads/ai" in path:
                return None                                                   # the bot branch doesn't exist yet
            if path.endswith("/git/commits/basesha"):
                return {"tree": {"sha": "basetree"}}
            if path.endswith("/git/blobs"):
                return {"sha": "blob1"}
            if path.endswith("/git/trees"):
                return {"sha": "tree1"}
            if path.endswith("/git/commits"):
                return {"sha": "commit1"}
            if path.endswith("/git/refs"):
                return {}
            if method == "GET" and path.endswith("/pulls"):
                return [{"number": 3, "html_url": "u", "title": "t", "head": {"ref": "ai/old", "repo": {"full_name": "me/app"}},
                         "base": {"ref": "main"}},
                        {"number": 2, "html_url": "u", "title": "t", "head": {"ref": "feature/x", "repo": {"full_name": "me/app"}},
                         "base": {"ref": "main"}}] if params and params.get("per_page") == 50 else []
            if method == "POST" and path.endswith("/pulls"):
                return {"number": 9, "html_url": PR_URL, "title": body["title"], "head": {"ref": body["head"]},
                        "base": {"ref": body["base"]}}
            raise AssertionError(f"unexpected request {method} {path}")
    return WireClient()


class ContractTests(unittest.TestCase):
    def test_fake_client_signatures_match_the_real_client(self):
        for name in ("view", "list_tree", "read_file", "latest_bot_branch", "commit_to_branch", "open_pr",
                     "sandbox_force_push"):
            fake = list(inspect.signature(getattr(FakeClient, name)).parameters.items())
            real = list(inspect.signature(getattr(gh.GitHubClient, name)).parameters.items())
            self.assertEqual([(n, p.default) for n, p in fake], [(n, p.default) for n, p in real], name)

    def test_the_fake_reports_a_missing_file_like_the_real_client(self):
        with self.assertRaises(gh.GitHubNotFound):
            FakeClient().read_file("me/app", "nope.py", "main")

    def test_repotarget_drives_the_real_client_over_a_fake_wire(self):
        """The real GitHubClient write path with only the HTTP layer replaced: catches any mismatch between what
        repotarget sends (changes, modes, branch, base) and what the client accepts."""
        calls = []
        client = make_wire_client(calls)
        self.assertEqual(client.latest_bot_branch("me/app"), "ai/old")           # only bot-prefixed PRs count
        text = job(Script([plan("src/app.py"), GOOD_EDIT]), client)
        self.assertIn(PR_URL, text)
        self.assertIn(("POST", "/repos/me/app/git/refs"), calls)                  # new branch was created


if __name__ == "__main__":
    unittest.main()
