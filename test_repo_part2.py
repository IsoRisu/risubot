# test_repo_part2.py -- offline tests. Run (from the bot folder): python -m unittest test_repo_part2 -v
import asyncio
import io
import json
import sys
import types
import unittest
import urllib.error

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

import llmclient  # noqa: E402
llmclient.VALIDATE_MODELS = False

import aitools  # noqa: E402
import githubclient as gh  # noqa: E402
import llmtools  # noqa: E402
import repoconfig as rc  # noqa: E402
import repocmds  # noqa: E402
import toolloop  # noqa: E402

GEMINI = next(p for p in llmclient._providers if p.name == "gemini")
GROQ = next(p for p in llmclient._providers if p.name == "groq")
TOOLS = aitools.tool_specs_for("sandbox")
HIST = [
    {"role": "user", "text": "make a file"},
    {"role": "assistant", "text": None, "calls": [{"id": "abc123def", "name": "write_file",
                                                    "args": {"path": "a.py", "content": "x=1"}, "_sig": "SIG"}]},
    {"role": "tool", "results": [{"id": "abc123def", "name": "write_file", "text": "OK", "is_error": False}]},
]


def http_error(code, body='{"error":{"message":"quota"}}'):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))


class BodyParseTests(unittest.TestCase):
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
        model_part = body["contents"][1]["parts"][0]
        self.assertEqual(model_part["thoughtSignature"], "SIG")
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
        turn = llmtools.chat_tools(HIST, TOOLS, "SYS")
        self.assertTrue(turn.backend.startswith("groq/"))
        self.assertEqual(turn.text, "from groq")

    def test_big_history_skips_small_providers(self):
        big = HIST + [{"role": "user", "text": "x" * 40_000}]
        GROQ._post = lambda *a: self.fail("groq must be skipped: its free prompt limit is ~30k chars")
        GEMINI._post = lambda model, idx, body, timeout: {"candidates": [{"content": {"parts": [{"text": "gem"}]}}]}
        self.assertTrue(llmtools.chat_tools(big, TOOLS, "SYS").backend.startswith("gemini/"))
        huge = HIST + [{"role": "user", "text": "x" * 250_000}]
        with self.assertRaises(rc.LimitExceeded):
            llmtools.chat_tools(huge, TOOLS, "SYS")

    def test_all_down_gives_unavailable(self):
        def fail(model, idx, body, timeout):
            raise http_error(429)
        GEMINI._post = GROQ._post = fail
        with self.assertRaises(llmclient.LLMUnavailable):
            llmtools.chat_tools(HIST, TOOLS, "SYS")


def scripted(turns):
    it = iter(turns)
    seen = []

    def chat(history, tools, system):
        seen.append(json.loads(json.dumps(history)))
        return next(it)
    chat.seen = seen
    return chat


def T(text="", calls=()):
    return llmtools.ToolTurn(text, [dict(id=f"id{i}{len(text)}", _sig=None, **c) for i, c in enumerate(calls)], "fake/m")


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

    def test_round_limit(self):
        chat = lambda h, t, s: T("", [dict(name="list_files", args={})])
        with self.assertRaises(rc.LimitExceeded):
            toolloop.run_tool_loop(aitools.ToolSession("sandbox"), "p", "SYS", chat=chat)

    def test_too_many_calls_in_round(self):
        calls = [dict(name="list_files", args={}) for _ in range(toolloop.MAX_CALLS_PER_ROUND + 2)]
        chat = scripted([T("", calls), T("ok")])
        toolloop.run_tool_loop(aitools.ToolSession("sandbox"), "p", "SYS", chat=chat)
        errs = [r["is_error"] for r in chat.seen[1][2]["results"]]
        self.assertEqual(errs[-2:], [True, True])

    def test_compaction_shrinks_old_writes_only(self):
        hist = [{"role": "user", "text": "p"},
                {"role": "assistant", "text": None, "calls": [{"id": "1", "name": "write_file", "_sig": None,
                                                                "args": {"path": "a", "content": "x" * 9000}}]},
                {"role": "tool", "results": [{"id": "1", "name": "write_file", "text": "OK", "is_error": False}]},
                {"role": "assistant", "text": None, "calls": [{"id": "2", "name": "write_file", "_sig": None,
                                                                "args": {"path": "b", "content": "y" * 9000}}]},
                {"role": "tool", "results": [{"id": "2", "name": "write_file", "text": "OK", "is_error": False}]}]
        size = toolloop.compact_history(hist, 12_000, 2_000)
        self.assertLessEqual(size, 12_000)
        self.assertIn("omitted", hist[1]["calls"][0]["args"]["content"])
        self.assertEqual(len(hist[3]["calls"][0]["args"]["content"]), 9000)      # recent one intact
        self.assertEqual(hist[1]["calls"][0]["args"]["path"], "a")


class FakeClient:
    def __init__(self):
        self.pushed = None

    def sandbox_force_push(self, files, message, branch=None):
        self.pushed = (dict(files), message, branch)
        return gh.CommitResult("c1", "https://github.com/me/sandbox/commit/c1", branch or "main")


class JobTests(unittest.TestCase):
    def test_parse_args(self):
        self.assertEqual(repocmds.parse_code_args("--branch demo make a game"), ("demo", "make a game"))
        self.assertEqual(repocmds.parse_code_args("--branch=demo x"), ("demo", "x"))
        self.assertEqual(repocmds.parse_code_args("make --branch x"), (None, "make --branch x"))
        with self.assertRaises(rc.ValidationError):
            repocmds.parse_code_args("--branch ../x go")

    def test_job_pushes_files(self):
        chat = scripted([T("", [dict(name="write_file", args={"path": "game/main.py", "content": "print('hi')"})]),
                         T("Wrote the game.")])
        client = FakeClient()
        msg = asyncio.run(repocmds.sandbox_job("make a game", "demo", client=client, chat=chat))
        self.assertEqual(client.pushed, ({"game/main.py": "print('hi')"}, "Sandbox output (1 file(s))", "demo"))
        self.assertIn("commit/c1", msg)
        self.assertIn("Wrote the game.", msg)

    def test_job_without_files_does_not_wipe(self):
        client = FakeClient()
        msg = asyncio.run(repocmds.sandbox_job("hi", None, client=client, chat=scripted([T("Hello!")])))
        self.assertIsNone(client.pushed)
        self.assertEqual(msg, "Hello!")

    def test_end_to_end_with_run_task_error_has_retry(self):
        import repotask

        def boom(history, tools, system):
            raise llmclient.LLMUnavailable("The AI is overloaded or out of quota right now. Try again in about 12 s.",
                                           retry_after=11)
        sent = []

        async def send(m):
            sent.append(m)

        async def go():
            task = repotask.TaskContext("sandbox", 42, 7, 1, "!code make x")
            return await repotask.run_task(task, send, lambda t: repocmds.sandbox_job("x", None, client=FakeClient(), chat=boom),
                                           lock=repotask.TaskLock())
        self.assertFalse(asyncio.run(go()))
        self.assertIn("Retry with: `!code make x`", sent[0])
        self.assertIn("12 s", sent[0])


if __name__ == "__main__":
    unittest.main()
