# test_repo_part4.py -- offline tests. Run (from the bot folder): python -m unittest test_repo_part4 -v
# Reuses the fakes of test_repo_part3 (imported as a module, so its tests are not collected twice).
import unittest
from types import SimpleNamespace

import test_repo_part3 as p3
import githubclient as gh
import llmclient
import repoconfig as rc
import repotarget

run, T, scripted, SPEC = p3.run, p3.T, p3.scripted, p3.SPEC
PR_URL = "https://github.com/me/app/pull/9"


class WSession(p3.FakeToolSession):
    def modes(self):
        return {p: "100755" for p in self.staged}


class WriteClient(p3.FakeClient):
    """FakeClient plus the Part 4 methods of the ASSUMED githubclient contract."""

    def __init__(self, files=None, latest=None, pr_error=None):
        super().__init__(files if files is not None else {"src/app.py": "def main(): ..."})
        self.commits, self.prs, self.latest, self.pr_error, self.latest_calls = [], [], latest, pr_error, []

    def latest_bot_branch(self, repo, prefix):
        self.latest_calls.append((repo, prefix))
        return self.latest

    def commit_to_branch(self, repo, **kw):
        self.commits.append((repo, kw))
        return gh.CommitResult("c1", "https://github.com/me/app/commit/c1", kw["branch"])

    def open_pr(self, repo, **kw):
        if self.pr_error:
            raise self.pr_error
        self.prs.append((repo, kw))
        return SimpleNamespace(url=PR_URL)


WRITE = [dict(name="write_file", args={"path": "src/new.py", "content": "x = 1\n"})]


def job(chat, client, prompt="add src/new.py", **kw):
    p3.FakeToolSession.instances.clear()
    return run(repotarget.target_job(SPEC, prompt, client=client, chat=chat, session_factory=WSession,
                                     read_only=False, **kw))


class NameTests(unittest.TestCase):
    def test_bot_branch_name_forces_prefix(self):
        self.assertEqual(repotarget.bot_branch_name("demo", "main"), "ai/demo")
        self.assertEqual(repotarget.bot_branch_name("ai/demo", "main"), "ai/demo")
        self.assertEqual(repotarget.bot_branch_name("main", "main"), "ai/main")      # can never hit the base
        with self.assertRaises(rc.ValidationError):
            repotarget.bot_branch_name("../x", "main")

    def test_new_branch_name(self):
        name = repotarget.new_branch_name("Fix the Login bug!! in src/app.py now", now=0)
        self.assertRegex(name, r"^ai/fix-the-login-bug-\d{4}-\d{6}$")
        self.assertRegex(repotarget.new_branch_name("???"), r"^ai/task-")

    def test_default_title_and_body(self):
        self.assertEqual(repotarget.default_title("\n\n  add   a   thing \nmore"), "add a thing")
        self.assertEqual(len(repotarget.default_title("x" * 300)), repotarget.DEFAULT_TITLE_CHARS)
        self.assertEqual(repotarget.default_title("   "), "AI changes")
        body = repotarget.default_body("ping @everyone", "done by @bob")
        self.assertNotIn("@everyone", body)
        self.assertNotIn("@bob", body)
        self.assertIn("AI summary", body)

    def test_retry_text_with_options(self):
        self.assertEqual(repotarget.retry_text("a/b", "do it", "dev", "ai/x", True, "T"),
                         "/repo task repo:a/b prompt:do it base:dev branch:ai/x iter:True title:T")


class WriteJobTests(unittest.TestCase):
    def test_commit_and_pr(self):
        client = WriteClient()
        chat = scripted([T("", WRITE), T("Added src/new.py.")])
        text = job(chat, client)
        (repo, kw), = client.commits
        self.assertEqual(repo, "me/app")
        self.assertEqual((kw["base"], kw["create"], kw["files"]), ("main", True, {"src/new.py": "x = 1\n"}))
        self.assertEqual(kw["modes"], {"src/new.py": "100755"})
        self.assertTrue(kw["branch"].startswith("ai/"))
        (_, pr), = client.prs
        self.assertEqual((pr["head"], pr["base"], pr["title"]), (kw["branch"], "main", "add src/new.py"))
        self.assertIn("Added src/new.py.", pr["body"])
        self.assertIn(PR_URL, text)
        self.assertIn("`src/new.py`", text)
        self.assertIn("Added src/new.py.", text)
        self.assertNotIn("read-only", text)
        first = chat.seen[0]["history"][0]["text"]
        self.assertNotIn("READ-ONLY", first)
        self.assertIn("stage edits", first)
        self.assertEqual(client.view_calls, [("me/app", "main")])

    def test_no_changes_means_no_commit(self):
        client = WriteClient()
        text = job(scripted([T("Nothing to change.")]), client)
        self.assertEqual((client.commits, client.prs), ([], []))
        self.assertIn("No changes", text)
        self.assertIn("Nothing to change.", text)
        self.assertNotIn("⚠️", text)

    def test_custom_branch_title_body(self):
        client = WriteClient()
        job(scripted([T("", WRITE), T("ok")]), client, branch="ai/demo", title="My title", body="Hello @team")
        self.assertEqual(client.commits[0][1]["branch"], "ai/demo")
        self.assertTrue(client.commits[0][1]["message"].startswith("My title"))
        pr = client.prs[0][1]
        self.assertEqual(pr["title"], "My title")
        self.assertNotIn("@team", pr["body"])
        self.assertTrue(pr["body"].startswith("Hello"))

    def test_pr_failure_keeps_the_branch_in_the_message(self):
        client = WriteClient(pr_error=RuntimeError("HTTP 422"))
        with self.assertLogs("repotarget", level="ERROR"):
            text = job(scripted([T("", WRITE), T("ok")]), client)
        self.assertEqual(len(client.commits), 1)
        self.assertIn("⚠️", text)
        self.assertIn(client.commits[0][1]["branch"], text)
        self.assertIn("HTTP 422", text)
        self.assertIn("commit/c1", text)

    def test_commit_failure_propagates(self):
        client = WriteClient()

        def boom(repo, **kw):
            raise rc.ValidationError("branch exists")
        client.commit_to_branch = boom
        with self.assertRaises(rc.ValidationError):
            job(scripted([T("", WRITE), T("ok")]), client)
        self.assertEqual(client.prs, [])

    def test_read_only_default_still_discards(self):
        client = WriteClient()
        p3.FakeToolSession.instances.clear()
        text = run(repotarget.target_job(SPEC, "x", client=client, chat=scripted([T("", WRITE), T("ok")]),
                                         session_factory=WSession))
        self.assertEqual(client.commits, [])
        self.assertIn("read-only", text)
        self.assertIn("⚠️", text)


class IterTests(unittest.TestCase):
    def test_iter_uses_newest_bot_branch(self):
        client = WriteClient(latest="ai/old-1")
        chat = scripted([T("", WRITE), T("done")])
        text = job(chat, client, iterate=True)
        self.assertEqual(client.latest_calls, [("me/app", "ai/")])
        self.assertEqual(client.view_calls, [("me/app", "ai/old-1")])
        kw = client.commits[0][1]
        self.assertEqual((kw["branch"], kw["create"], kw["base"]), ("ai/old-1", False, "main"))
        self.assertEqual(client.prs[0][1]["head"], "ai/old-1")
        self.assertIn("ai/old-1", chat.seen[0]["history"][0]["text"])
        self.assertIn("(continued)", text)

    def test_iter_with_named_branch_skips_lookup(self):
        client = WriteClient(latest="ai/other")
        job(scripted([T("", WRITE), T("ok")]), client, iterate=True, branch="ai/mine")
        self.assertEqual(client.latest_calls, [])
        self.assertEqual(client.view_calls, [("me/app", "ai/mine")])

    def test_iter_without_any_bot_branch_is_a_validation_error(self):
        client = WriteClient(latest=None)
        with self.assertRaises(rc.ValidationError):
            job(scripted([]), client, iterate=True)
        self.assertEqual(client.view_calls, [])


class SlashTests(unittest.TestCase):
    def command(self, chat, client):
        group = repotarget.register(p3.FakeBot(), client_factory=lambda: client, chat=chat,
                                    session_factory=WSession, sync=False)
        return group.get_command("task").callback

    def invoke(self, chat, client, **args):
        it = p3.FakeInteraction()
        args = {"repo": "me/app", "prompt": "add src/new.py", "base": None, "branch": None, "iter": False,
                "title": None, "body": None, **args}
        run(self.command(chat, client)(it, **args))
        return it

    def test_write_run_posts_pr_link(self):
        client = WriteClient()
        it = self.invoke(scripted([T("", WRITE), T("Added it.")]), client, branch="demo", title="  Add new  ")
        self.assertEqual(it.kinds()[0], "defer")
        self.assertTrue(any(PR_URL in e[1] for e in it.ev if e[0] == "followup"))
        self.assertEqual(client.commits[0][1]["branch"], "ai/demo")
        self.assertEqual(client.prs[0][1]["title"], "Add new")

    def test_iter_flag_reaches_the_job(self):
        client = WriteClient(latest="ai/old-1")
        self.invoke(scripted([T("", WRITE), T("ok")]), client, iter=True)
        self.assertEqual(client.view_calls, [("me/app", "ai/old-1")])

    def test_iter_failure_reports_with_retry(self):
        client = WriteClient(latest=None)
        it = self.invoke(scripted([]), client, iter=True)
        self.assertEqual(it.kinds()[0], "defer")
        self.assertIn("no bot branch", it.ev[1][1].lower())
        self.assertIn("iter:True", it.ev[1][1])

    def assert_refused(self, it, client):
        self.assertEqual(it.kinds(), ["response"])
        self.assertTrue(it.ev[0][2]["ephemeral"])
        self.assertEqual((client.view_calls, client.commits), ([], []))

    def test_bad_branch_title_body_refused_before_defer(self):
        for bad in ({"branch": "../x"}, {"title": "t" * (repotarget.MAX_TITLE_CHARS + 1)},
                    {"body": "b" * (repotarget.MAX_BODY_CHARS + 1)}):
            client = WriteClient()
            self.assert_refused(self.invoke(scripted([]), client, **bad), client)

    def test_llm_outage_still_reports(self):
        client = WriteClient()

        def boom(h, t, s):
            raise llmclient.LLMUnavailable("The AI is overloaded or out of quota right now. Try again in about 12 s.",
                                           retry_after=11)
        it = self.invoke(boom, client, branch="demo", iter=False)
        self.assertIn("Retry with: `/repo task repo:me/app prompt:add src/new.py branch:demo`", it.ev[1][1])


if __name__ == "__main__":
    unittest.main()
