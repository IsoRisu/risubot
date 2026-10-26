# test_bridge.py -- offline tests for llmbridge (Option A: local tool loop + remote finalizer).
# Run from the bot folder: python -m unittest test_bridge -v
#
# Reuses the fakes and setup of test_repo.py (imported as a module, so its own tests are not
# collected twice). If you run `python -m unittest discover`, test_repo's job tests will also
# run; two of them (`test_job_pushes_files_but_empty_run_does_not_wipe` and
# `test_job_without_files_does_not_wipe`) expect a plain local text back and will now see the
# "⚠️ Remote AI unavailable" marker when there is no network. Patch the finalize in those
# two tests (no production code change needed):
#
#     from unittest.mock import patch
#     import llmbridge
#     with patch.object(llmbridge, "_default_remote_finalize", side_effect=lambda p, t, c: t):
#         msg = run(repocmds.sandbox_job(...))

import unittest
from unittest.mock import patch

import test_repo as tr          # sets up botkey + discord stub, imports the modules under test

import aitools
import llmbridge
import llmclient
import llmtools
import repoconfig as rc
import repotarget

run, T, scripted = tr.run, tr.T, tr.scripted
FakeClient, FakeToolSession = tr.FakeClient, tr.FakeToolSession
SPEC, WRITE, PR_URL = tr.SPEC, tr.WRITE, tr.PR_URL


# ---------------------------------------------------------------------------
# Bridge: normal path, outages, propagation
# ---------------------------------------------------------------------------
class BridgeTests(unittest.TestCase):
    def test_happy_path_uses_both_backends(self):
        local_chat = scripted([T("", WRITE), T("Wrote src/new.py.")])
        remote_calls = []

        def finalize(prompt, local_text, changes):
            remote_calls.append((prompt, local_text, dict(changes)))
            return "Polished final reply."

        session = aitools.ToolSession("sandbox")
        with patch.object(llmclient, "last_backend", return_value="gemini/gemini-3-flash"):
            result = llmbridge.run_bridged_loop(
                session, "add src/new.py", "SYS",
                local_chat=local_chat, remote_finalize=finalize)
        self.assertEqual(result.text, "Polished final reply.")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.backend, "fake/m → gemini/gemini-3-flash")
        self.assertEqual(remote_calls,
                         [("add src/new.py", "Wrote src/new.py.", {"src/new.py": "x = 1\n"})])
        self.assertEqual(session.changes(), {"src/new.py": "x = 1\n"})

    def test_outage_llm_unavailable_falls_back(self):
        local_chat = scripted([T("Wrote src/new.py.")])

        def boom(prompt, local_text, changes):
            raise llmclient.LLMUnavailable("quota exceeded")

        session = aitools.ToolSession("sandbox")
        result = llmbridge.run_bridged_loop(
            session, "p", "SYS", local_chat=local_chat, remote_finalize=boom)
        self.assertTrue(result.text.startswith("⚠️ Remote AI unavailable"))
        self.assertIn("quota exceeded", result.text)
        self.assertIn("Wrote src/new.py.", result.text)
        self.assertNotIn("→", result.backend)  # no remote backend when the finalizer failed

    def test_outage_llm_keys_rejected_falls_back(self):
        local_chat = scripted([T("Local answer.")])

        def boom(prompt, local_text, changes):
            raise llmclient.LLMKeysRejected("all keys rejected")

        session = aitools.ToolSession("sandbox")
        result = llmbridge.run_bridged_loop(
            session, "p", "SYS", local_chat=local_chat, remote_finalize=boom)
        self.assertTrue(result.text.startswith("⚠️ Remote AI unavailable"))
        self.assertIn("keys rejected", result.text)
        self.assertIn("Local answer.", result.text)

    def test_local_failure_propagates_without_finalize(self):
        calls = []

        def boom_local(history, tools, system):
            raise llmclient.LLMUnavailable("local down")

        def finalize(prompt, local_text, changes):
            calls.append(1)
            return "should not happen"

        session = aitools.ToolSession("sandbox")
        with self.assertRaises(llmclient.LLMUnavailable):
            llmbridge.run_bridged_loop(session, "p", "SYS",
                                       local_chat=boom_local, remote_finalize=finalize)
        self.assertEqual(calls, [])

    def test_local_limit_exceeded_propagates(self):
        def boom_local(history, tools, system):
            raise rc.LimitExceeded("too big for local model")

        session = aitools.ToolSession("sandbox")
        with self.assertRaises(rc.LimitExceeded):
            llmbridge.run_bridged_loop(session, "p", "SYS",
                                       local_chat=boom_local,
                                       remote_finalize=lambda p, t, c: "no")


# ---------------------------------------------------------------------------
# Read-only target run still goes through the finalizer
# ---------------------------------------------------------------------------
class ReadOnlyTargetTests(unittest.TestCase):
    def test_read_only_target_still_finalizes_and_commits_nothing(self):
        client = FakeClient()
        # Would-be edits are staged by the loop but read_only=True discards them.
        local_chat = scripted([T("", WRITE), T("ok")])
        with patch.object(llmclient, "ask", return_value="Polished read-only answer") as mock_ask:
            with patch.object(llmclient, "last_backend", return_value="remote/backend"):
                result = repotarget.run_target(
                    SPEC, "what does src/app.py do?",
                    client=client, chat=local_chat,
                    session_factory=FakeToolSession, read_only=True)
        self.assertEqual(result.result.text, "Polished read-only answer")
        self.assertEqual(client.commits, [])
        self.assertEqual(client.prs, [])
        self.assertTrue(result.read_only)
        # The finalizer used the compact prompt, not the ~11k default.
        self.assertEqual(mock_ask.call_args.kwargs.get("system_prompt"),
                         llmbridge.FINALIZE_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# llmclient.ask(system_prompt=...) override
# ---------------------------------------------------------------------------
class AskSystemPromptTests(unittest.TestCase):
    def _captured_prompt(self, **ask_kwargs):
        captured = []

        def fake_generate(prompt, attachments, system_prompt):
            captured.append(system_prompt)
            return "ok"

        with patch.object(llmclient, "_generate", new=fake_generate):
            llmclient.ask("hi", **ask_kwargs)
        return captured[-1]

    def test_system_prompt_overrides(self):
        self.assertEqual(self._captured_prompt(system_prompt="CUSTOM"), "CUSTOM")

    def test_guild_context_still_appends(self):
        sp = self._captured_prompt(guild_context="GUILD MEMORY")
        self.assertIn("GUILD MEMORY", sp)
        self.assertIn(llmclient.SYSTEM_PROMPT, sp)

    def test_default_is_the_big_prompt(self):
        self.assertEqual(self._captured_prompt(), llmclient.SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# llmtools.chat_tools_local without a configured local model
# ---------------------------------------------------------------------------
class LocalUnconfiguredTests(unittest.TestCase):
    def test_chat_tools_local_raises_when_unconfigured(self):
        # The botkey stub has no LOCAL_MODEL, so local_provider() is None.
        self.assertIsNone(llmclient.local_provider())
        with self.assertRaises(llmclient.LLMUnavailable) as cm:
            llmtools.chat_tools_local([{"role": "user", "text": "hi"}], [], "SYS")
        self.assertIn("LOCAL_MODEL", str(cm.exception))


if __name__ == "__main__":
    unittest.main()