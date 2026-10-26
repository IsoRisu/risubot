# test_repo.py -- offline tests for the repo/AI-coding feature. Run from the bot folder:
#     python -m unittest test_repo -v
# No network, no real keys: botkey is replaced by a stub, and discord is stubbed when not installed.
import asyncio
import inspect
import io
import json
import sys
import types
import unittest
import urllib.error
from types import SimpleNamespace

bk = types.ModuleType("botkey")
bk.bot_key = "DISCORDSECRET.aaaaaa.bbbbbbbbbbbbbbbbbbbbbbbbbbbb"
bk.GITHUB_TOKEN = "github_pat_" + "A" * 50
bk.GEMINI_API_KEY = ["AIza" + "B" * 35]
bk.GROQ_API_KEY = "gsk_" + "C" * 40
bk.REPO_OWNER_IDS = [42]
bk.REPO_ALLOWED_GUILD_IDS = [7]
bk.SANDBOX_REPO = "me/sandbox"
bk.REPO_ALLOWLIST = {"me/app": ["main"]}
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

import aitools  # noqa: E402
import githubclient as gh  # noqa: E402
import llmclient  # noqa: E402
import llmtools  # noqa: E402
import repocmds  # noqa: E402
import repoconfig as rc  # noqa: E402
import repotarget  # noqa: E402
import repotask  # noqa: E402
import toolloop  # noqa: E402

llmclient.VALIDATE_MODELS = False
GEMINI = next(p for p in llmclient._providers if p.name == "gemini")
GROQ = next(p for p in llmclient._providers if p.name == "groq")
TOOLS = aitools.tool_specs_for("sandbox")
SPEC = rc.resolve_target("me/app")
PR_URL = "https://github.com/me/app/pull/9"
OVERLOADED = llmclient.LLMUnavailable("The AI is overloaded or out of quota right now. Try again in about 12 s.",
                                      retry_after=11)
HIST = [
    {"role": "user", "text": "make a file"},
    {"role": "assistant", "text": None, "calls": [{"id": "abc123def", "name": "write_file",
                                                    "args": {"path": "a.py", "content": "x=1"}, "_sig": "SIG"}]},
    {"role": "tool", "results": [{"id": "abc123def", "name": "write_file", "text": "OK", "is_error": False}]},
]
WRITE = [dict(name="write_file", args={"path": "src/new.py", "content": "x = 1\n"})]


# ---------------------------------------------------------------------------
# helpers and fakes
# ---------------------------------------------------------------------------
def run(coro):
    return asyncio.run(coro)


def http_error(code, body='{"error":{"message":"quota"}}'):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))


def scripted(turns):
    """A fake `chat` that replays `turns` and records the history it was shown."""
    it = iter(turns)
    seen = []

    def chat(history, tools, system):
        seen.append(json.loads(json.dumps(history)))
        return next(it)
    chat.seen = seen
    return chat


def T(text="", calls=()):
    return llmtools.ToolTurn(text, [dict(id=f"id{i}{len(text)}", _sig=None, **c) for i, c in enumerate(calls)], "fake/m")


class FakeToolSession(aitools.ToolSession):
    """Real ToolSession that also records itself; staged edits are exposed as .staged."""
    instances = []

    def __init__(self, mode, view=None):
        super().__init__(mode, view)
        FakeToolSession.instances.append(self)

    @property
    def staged(self):
        return self.changes()

    def modes(self):
        return {p: "100755" for p in self.staged}


class FakeClient:
    """Stands in for GitHubClient. Reads come from `files`; writes are recorded. The signatures of
    everything repotarget/repocmds call must match the real client (see ContractTests)."""

    def __init__(self, files=None, latest=None, pr_error=None):
        self.files = files if files is not None else {"src/app.py": "def main(): ..."}
        self.latest, self.pr_error = latest, pr_error
        self.view_calls, self.latest_calls, self.commits, self.prs, self.pushed = [], [], [], [], None

    def view(self, repo, ref):
        self.view_calls.append((repo, ref))
        return gh.RepoView(self, repo, ref)

    def list_tree(self, repo, ref):
        return gh.TreeListing("sha", [gh.TreeEntry(p, len(t), "100644", "s") for p, t in self.files.items()])

    def read_file(self, repo, path, ref):
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


def job(chat, client, prompt="add src/new.py", **kw):
    FakeToolSession.instances.clear()
    kw.setdefault("read_only", False)
    return run(repotarget.target_job(SPEC, prompt, client=client, chat=chat, session_factory=FakeToolSession, **kw))


# ---------------------------------------------------------------------------
# LLM wire formats and provider fallback
# ---------------------------------------------------------------------------
class LlmToolsTests(unittest.TestCase):
    def test_openai_roundtrip(self):
        body = json.loads(llmtools.build_body(GROQ, "m", HIST, TOOLS, "SYS"))
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SYS"})
        asst, tool = body["messages"][2], body["messages"][3]
        self.assertEqual(asst["tool_calls"][0]["id"], "abc123def")
        self.assertEqual(json.loads(asst["tool_calls"][0]["function"]["arguments"])["path"], "a.py")
        self.assertEqual((tool["role"], tool["tool_call_id"]), ("tool", "abc123def"))
        self.assertEqual(body["tools"][0]["type"], "function")
        payload = {"choices": [{"message": {"content": "<think>x</think>hi", "tool_calls": [
            {"id": "zzz", "function": {"name": "write_file", "arguments": '{"path":"b","content":"c"}'}},
            {"id": "yyy", "function": {"name": "read_file", "arguments": "{bad json"}}]}}]}
        turn = llmtools.parse_response(GROQ, payload, "m")
        self.assertEqual(turn.text, "hi")
        self.assertEqual(turn.calls[0]["args"], {"path": "b", "content": "c"})
        self.assertIsNone(turn.calls[1]["args"])
        self.assertEqual(len(turn.calls[0]["id"]), 9)

    def test_gemini_roundtrip_and_signatures(self):
        body = json.loads(llmtools.build_body(GEMINI, "gemini-3-flash", HIST, TOOLS, "SYS"))
        self.assertEqual(body["contents"][1]["parts"][0]["thoughtSignature"], "SIG")
        self.assertEqual(body["contents"][2]["parts"][0]["functionResponse"]["response"], {"result": "OK"})
        self.assertEqual(body["tools"][0]["functionDeclarations"][0]["name"], "list_files")
        nosig = json.loads(json.dumps(HIST))
        nosig[1]["calls"][0]["_sig"] = None
        b2 = json.loads(llmtools.build_body(GEMINI, "gemini-3-flash", nosig, TOOLS, "S"))
        self.assertEqual(b2["contents"][1]["parts"][0]["thoughtSignature"], llmtools.GEMINI_SKIP_SIGNATURE)
        b3 = json.loads(llmtools.build_body(GEMINI, "gemini-2.5-flash", nosig, TOOLS, "S"))
        self.assertNotIn("thoughtSignature", b3["contents"][1]["parts"][0])
        payload = {"candidates": [{"content": {"parts": [
            {"text": "thinking", "thought": True}, {"text": "ok"},
            {"functionCall": {"name": "write_file", "args": {"path": "p", "content": "c"}}, "thoughtSignature": "S2"}]}}]}
        turn = llmtools.parse_response(GEMINI, payload, "gemini-3-flash")
        self.assertEqual((turn.text, turn.calls[0]["_sig"]), ("ok", "S2"))
        with self.assertRaises(llmclient.LLMBlocked):
            llmtools.parse_response(GEMINI, {"candidates": [{"finishReason": "SAFETY"}]}, "m")


class FallbackTests(unittest.TestCase):
    def setUp(self):
        for p in (GEMINI, GROQ):
            p.pool = llmclient._KeyPool(len(p.keys))
        self._orig = (GEMINI._post, GROQ._post)

    def tearDown(self):
        GEMINI._post, GROQ._post = self._orig

    def test_falls_back_to_groq_on_429(self):
        def g_fail(model, idx, body, timeout):
            raise http_error(429)
        GEMINI._post = g_fail
        GROQ._post = lambda model, idx, body, timeout: {"choices": [{"message": {"content": "from groq"}}]}
        with _quiet():
            turn = llmtools.chat_tools(HIST, TOOLS, "SYS")
        self.assertTrue(turn.backend.startswith("groq/"))
        self.assertEqual(turn.text, "from groq")

    def test_big_history_skips_small_providers(self):
        big = HIST + [{"role": "user", "text": "x" * 40_000}]
        GROQ._post = lambda *a: self.fail("groq must be skipped: its free prompt limit is ~30k chars")
        GEMINI._post = lambda model, idx, body, timeout: {"candidates": [{"content": {"parts": [{"text": "gem"}]}}]}
        self.assertTrue(llmtools.chat_tools(big, TOOLS, "SYS").backend.startswith("gemini/"))
        with self.assertRaises(rc.LimitExceeded):
            llmtools.chat_tools(HIST + [{"role": "user", "text": "x" * 250_000}], TOOLS, "SYS")

    def test_all_down_gives_unavailable(self):
        def fail(model, idx, body, timeout):
            raise http_error(429)
        GEMINI._post = GROQ._post = fail
        with _quiet(), self.assertRaises(llmclient.LLMUnavailable):
            llmtools.chat_tools(HIST, TOOLS, "SYS")


class _quiet:
    """Silence the providers' cooldown chatter (they print) while a test provokes it."""

    def __enter__(self):
        self._out, sys.stdout = sys.stdout, io.StringIO()

    def __exit__(self, *exc):
        sys.stdout = self._out


# ---------------------------------------------------------------------------
# tool loop
# ---------------------------------------------------------------------------
class LoopTests(unittest.TestCase):
    def test_write_then_finish(self):
        chat = scripted([
            T("", [dict(name="write_file", args={"path": "a.py", "content": "print(1)\n"}),
                   dict(name="write_file", args={"path": "../evil", "content": "x"}),
                   dict(name="read_file", args=None)]),
            T("Done!"),
        ])
        s = aitools.ToolSession("sandbox")
        res = toolloop.run_tool_loop(s, "make a.py", "SYS", chat=chat)
        self.assertEqual((res.text, res.rounds), ("Done!", 2))
        self.assertEqual(s.changes(), {"a.py": "print(1)\n"})
        results = chat.seen[1][2]["results"]
        self.assertEqual([r["is_error"] for r in results], [False, True, True])
        self.assertIn("not valid JSON", results[2]["text"])

    def test_limits(self):
        with self.assertRaises(rc.LimitExceeded):          # endless tool calls hit the round limit
            toolloop.run_tool_loop(aitools.ToolSession("sandbox"), "p", "SYS",
                                   chat=lambda h, t, s: T("", [dict(name="list_files", args={})]))
        calls = [dict(name="list_files", args={}) for _ in range(toolloop.MAX_CALLS_PER_ROUND + 2)]
        chat = scripted([T("", calls), T("ok")])           # calls beyond the per-round cap are refused
        toolloop.run_tool_loop(aitools.ToolSession("sandbox"), "p", "SYS", chat=chat)
        self.assertEqual([r["is_error"] for r in chat.seen[1][2]["results"]][-2:], [True, True])

    def test_compaction_shrinks_old_writes_only(self):
        def write(i, ch):
            return [{"role": "assistant", "text": None, "calls": [{"id": i, "name": "write_file", "_sig": None,
                                                                    "args": {"path": i, "content": ch * 9000}}]},
                    {"role": "tool", "results": [{"id": i, "name": "write_file", "text": "OK", "is_error": False}]}]
        hist = [{"role": "user", "text": "p"}] + write("a", "x") + write("b", "y")
        self.assertLessEqual(toolloop.compact_history(hist, 12_000, 2_000), 12_000)
        self.assertIn("omitted", hist[1]["calls"][0]["args"]["content"])
        self.assertEqual(len(hist[3]["calls"][0]["args"]["content"]), 9000)      # the recent write is intact
        self.assertEqual(hist[1]["calls"][0]["args"]["path"], "a")


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

    def test_job_pushes_files_but_empty_run_does_not_wipe(self):
        chat = scripted([T("", [dict(name="write_file", args={"path": "game/main.py", "content": "print('hi')"})]),
                         T("Wrote the game.")])
        client = FakeClient()
        msg = run(repocmds.sandbox_job("make a game", "demo", client=client, chat=chat))
        self.assertEqual(client.pushed, ({"game/main.py": "print('hi')"}, "Sandbox output (1 file(s))", "demo"))
        self.assertIn("commit/c1", msg)
        self.assertIn("Wrote the game.", msg)
        client = FakeClient()
        self.assertEqual(run(repocmds.sandbox_job("hi", None, client=client, chat=scripted([T("Hello!")]))), "Hello!")
        self.assertIsNone(client.pushed)

    def test_outage_error_has_retry_command(self):
        sent = []

        async def send(m):
            sent.append(m)

        def boom(history, tools, system):
            raise OVERLOADED

        async def go():
            task = repotask.TaskContext("sandbox", 42, 7, 1, "!code make x")
            return await repotask.run_task(task, send, lambda t: repocmds.sandbox_job("x", None, client=FakeClient(), chat=boom),
                                           lock=repotask.TaskLock())
        self.assertFalse(run(go()))
        self.assertIn("Retry with: `!code make x`", sent[0])
        self.assertIn("12 s", sent[0])


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


class WriteJobTests(unittest.TestCase):
    def test_commit_and_pr(self):
        client = FakeClient()
        chat = scripted([T("", WRITE), T("Added src/new.py.")])
        text = job(chat, client)
        (repo, kw), = client.commits
        self.assertEqual(repo, "me/app")
        self.assertEqual((kw["base"], kw["changes"]), ("main", {"src/new.py": "x = 1\n"}))
        self.assertEqual(kw["modes"], {"src/new.py": "100755"})
        self.assertTrue(kw["branch"].startswith("ai/"))
        (_, pr), = client.prs
        self.assertEqual((pr["head"], pr["base"], pr["title"]), (kw["branch"], "main", "add src/new.py"))
        self.assertIn("Added src/new.py.", pr["body"])
        for part in (PR_URL, "`src/new.py`", "Added src/new.py."):
            self.assertIn(part, text)
        self.assertNotIn("read-only", text)
        first = chat.seen[0][0]["text"]
        self.assertNotIn("READ-ONLY", first)
        self.assertIn("stage edits", first)
        self.assertEqual(client.view_calls, [("me/app", "main")])

    def test_no_changes_means_no_commit(self):
        client = FakeClient()
        text = job(scripted([T("Nothing to change.")]), client)
        self.assertEqual((client.commits, client.prs), ([], []))
        self.assertIn("No changes", text)
        self.assertNotIn("⚠️", text)

    def test_custom_branch_title_body(self):
        client = FakeClient()
        job(scripted([T("", WRITE), T("ok")]), client, branch="ai/demo", title="My title", body="Hello @team")
        self.assertEqual(client.commits[0][1]["branch"], "ai/demo")
        self.assertTrue(client.commits[0][1]["message"].startswith("My title"))
        pr = client.prs[0][1]
        self.assertEqual(pr["title"], "My title")
        self.assertNotIn("@team", pr["body"])
        self.assertTrue(pr["body"].startswith("Hello"))

    def test_pr_failure_keeps_the_branch_in_the_message(self):
        client = FakeClient(pr_error=RuntimeError("HTTP 422"))
        with self.assertLogs("repotarget", level="ERROR"):
            text = job(scripted([T("", WRITE), T("ok")]), client)
        self.assertEqual(len(client.commits), 1)
        for part in ("⚠️", client.commits[0][1]["branch"], "HTTP 422", "commit/c1"):
            self.assertIn(part, text)

    def test_commit_failure_propagates_and_opens_no_pr(self):
        client = FakeClient()

        def boom(repo, **kw):
            raise rc.ValidationError("branch exists")
        client.commit_to_branch = boom
        with self.assertRaises(rc.ValidationError):
            job(scripted([T("", WRITE), T("ok")]), client)
        self.assertEqual(client.prs, [])

    def test_read_only_default_discards_edits(self):
        client = FakeClient()
        text = job(scripted([T("", WRITE), T("ok")]), client, read_only=True)
        self.assertEqual(client.commits, [])
        self.assertIn("read-only", text)
        self.assertIn("⚠️", text)


class IterTests(unittest.TestCase):
    def test_iter_uses_newest_bot_branch(self):
        client = FakeClient(latest="ai/old-1")
        chat = scripted([T("", WRITE), T("done")])
        text = job(chat, client, iterate=True)
        self.assertEqual(client.latest_calls, ["me/app"])
        self.assertEqual(client.view_calls, [("me/app", "ai/old-1")])
        kw = client.commits[0][1]
        self.assertEqual((kw["branch"], kw["base"]), ("ai/old-1", "main"))
        self.assertEqual(client.prs[0][1]["head"], "ai/old-1")
        self.assertIn("ai/old-1", chat.seen[0][0]["text"])
        self.assertIn("(continued)", text)

    def test_iter_with_named_branch_skips_lookup(self):
        client = FakeClient(latest="ai/other")
        job(scripted([T("", WRITE), T("ok")]), client, iterate=True, branch="ai/mine")
        self.assertEqual(client.latest_calls, [])
        self.assertEqual(client.view_calls, [("me/app", "ai/mine")])

    def test_iter_without_any_bot_branch_is_a_validation_error(self):
        client = FakeClient(latest=None)
        with self.assertRaises(rc.ValidationError):
            job(scripted([]), client, iterate=True)
        self.assertEqual(client.view_calls, [])


class SlashTests(unittest.TestCase):
    def invoke(self, chat, client, **args):
        group = repotarget.register(FakeBot(), client_factory=lambda: client, chat=chat,
                                    session_factory=FakeToolSession, sync=False)
        it = FakeInteraction()
        args = {"repo": "me/app", "prompt": "add src/new.py", "base": None, "branch": None, "iter": False,
                "title": None, "body": None, **args}
        FakeToolSession.instances.clear()
        run(group.get_command("task").callback(it, **args))
        return it

    def test_write_run_posts_pr_link(self):
        client = FakeClient()
        it = self.invoke(scripted([T("", WRITE), T("Added it.")]), client, branch="demo", title="  Add new  ")
        self.assertEqual(it.kinds()[0], "defer")
        self.assertTrue(any(PR_URL in e[1] for e in it.ev if e[0] == "followup"))
        self.assertEqual(client.commits[0][1]["branch"], "ai/demo")
        self.assertEqual(client.prs[0][1]["title"], "Add new")

    def test_iter_flag_reaches_the_job_and_failure_has_retry(self):
        client = FakeClient(latest="ai/old-1")
        self.invoke(scripted([T("", WRITE), T("ok")]), client, iter=True)
        self.assertEqual(client.view_calls, [("me/app", "ai/old-1")])
        it = self.invoke(scripted([]), FakeClient(latest=None), iter=True)
        self.assertEqual(it.kinds()[0], "defer")
        self.assertIn("no bot branch", it.ev[1][1].lower())
        self.assertIn("iter:True", it.ev[1][1])

    def test_bad_input_is_refused_before_defer(self):
        too_long = {"title": "t" * (repotarget.MAX_TITLE_CHARS + 1)}, {"body": "b" * (repotarget.MAX_BODY_CHARS + 1)}
        for bad in ({"branch": "../x"}, *too_long):
            client = FakeClient()
            it = self.invoke(scripted([]), client, **bad)
            self.assertEqual(it.kinds(), ["response"])
            self.assertTrue(it.ev[0][2]["ephemeral"])
            self.assertEqual((client.view_calls, client.commits), ([], []))

    def test_llm_outage_still_reports(self):
        def boom(h, t, s):
            raise OVERLOADED
        it = self.invoke(boom, FakeClient(), branch="demo")
        self.assertIn("Retry with: `/repo task repo:me/app prompt:add src/new.py branch:demo`", it.ev[1][1])


# ---------------------------------------------------------------------------
# the fake client must not drift from the real one
# ---------------------------------------------------------------------------
class ContractTests(unittest.TestCase):
    def test_fake_client_signatures_match_the_real_client(self):
        for name in ("view", "list_tree", "read_file", "latest_bot_branch", "commit_to_branch", "open_pr",
                     "sandbox_force_push"):
            fake = list(inspect.signature(getattr(FakeClient, name)).parameters.items())
            real = list(inspect.signature(getattr(gh.GitHubClient, name)).parameters.items())
            self.assertEqual([(n, p.default) for n, p in fake], [(n, p.default) for n, p in real], name)

    def test_repotarget_drives_the_real_client_over_a_fake_wire(self):
        """The real GitHubClient write path (commit_to_branch, open_pr, latest_bot_branch) with only the HTTP
        layer replaced: catches any mismatch between what repotarget sends and what the client accepts."""
        calls = []

        class WireClient(gh.GitHubClient):          # real write path; reads served from memory
            view, list_tree, read_file = FakeClient.view, FakeClient.list_tree, FakeClient.read_file

            def __init__(self):
                super().__init__(token="t")
                self.files, self.view_calls = {"src/app.py": "def main(): ..."}, []

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

        client = WireClient()
        self.assertEqual(client.latest_bot_branch("me/app"), "ai/old")           # only bot-prefixed PRs count
        text = job(scripted([T("", WRITE), T("ok")]), client)
        self.assertIn(PR_URL, text)
        self.assertIn(("POST", "/repos/me/app/git/refs"), calls)                  # new branch was created


if __name__ == "__main__":
    unittest.main()
