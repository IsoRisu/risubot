# githubclient.py
# Blocking GitHub REST wrapper (call it from async code with asyncio.to_thread).
#
# Two entry points that write, and they are deliberately different:
#   Sandbox (hardcoded repo, NO repo argument):  sandbox_force_push(files, message)
#   Target  (allowlisted repo, bot branches):    commit_to_branch(...), open_pr(...), update_pr(...)
# Reads for target repos: list_tree(), read_file(), view() -> RepoView.
#
# There is NO merge capability: _request refuses merge endpoints and the class has no merge method.
# Force-updating a ref is only possible for the sandbox repo.

from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import quote

import requests

import repoconfig as rc
from repoconfig import LimitExceeded, RepoBotError, SecretLeakError, ValidationError

log = logging.getLogger("githubclient")

API = "https://api.github.com"
_ALLOWED_METHODS = frozenset({"GET", "POST", "PATCH", "PUT"})        # never DELETE
_MERGE_RE = re.compile(r"^/repos/[^/]+/[^/]+/(?:pulls/\d+/merge|merges)(?:/|$|\?)")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class GitHubError(RepoBotError):
    def __init__(self, message, *, status=None, retry_after=None):
        super().__init__(message, retry_after=retry_after)
        self.status = status


class GitHubAuthError(GitHubError):
    show_retry = False


class GitHubNotFound(GitHubError):
    pass


class GitHubRateLimit(GitHubError):
    pass


class GitHubConflict(GitHubError):
    pass


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TreeEntry:
    path: str
    size: int
    mode: str
    sha: str


@dataclass
class TreeListing:
    commit_sha: str
    entries: list
    truncated: bool = False          # GitHub cut the listing (huge repo)

    def render(self, max_chars: int = rc.MAX_TREE_CONTEXT_CHARS) -> str:
        """Compact tree for the prompt. Full list if it fits, else folded by directory, else cut."""
        files = [e for e in self.entries if not rc.is_ignored_path(e.path)]
        lines = [f"{e.path} ({_human(e.size)})" if e.size >= 10_000 else e.path for e in sorted(files, key=lambda e: e.path)]
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text + ("\n[listing truncated by GitHub]" if self.truncated else "")
        groups: dict = {}
        for e in files:
            parts = e.path.split("/")
            key = "/".join(parts[:2]) + "/" if len(parts) > 2 else (parts[0] + "/" if len(parts) == 2 else "")
            groups.setdefault(key, []).append(e.path)
        out = []
        for key in sorted(groups):
            paths = groups[key]
            out.extend(paths if key == "" else [f"{key} ({len(paths)} files)"])
        text = "\n".join(out)
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0] + "\n[... tree cut; use list_files(directory) ...]"
        return "[tree folded: use list_files(directory) for details]\n" + text


@dataclass
class FileContent:
    path: str
    text: str
    size: int
    sha: str
    is_binary: bool = False


@dataclass
class CommitResult:
    sha: str
    url: str
    branch: str


@dataclass
class PullRequest:
    number: int
    url: str
    head: str
    base: str
    title: str = ""
    reused: bool = False             # an open PR for this head branch already existed


def _human(n: int) -> str:
    return f"{n / 1000:.0f}k" if n < 1_000_000 else f"{n / 1_000_000:.1f}M"


def _q(path: str) -> str:
    return quote(path, safe="/")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class GitHubClient:
    def __init__(self, token: str | None = None, session=None):
        self._token_override = token
        self.session = session or requests.Session()
        self._sleep = time.sleep           # replaced in tests

    # -- plumbing --------------------------------------------------------
    def _token(self) -> str:
        return self._token_override or rc.require_github_token()

    def _request(self, method, path, *, body=None, params=None, accept=None, ok=(200, 201), not_found_ok=False):
        if method not in _ALLOWED_METHODS or _MERGE_RE.match(path):
            raise RepoBotError("Blocked GitHub request (bot policy: no merging, no deleting).")
        headers = {"Authorization": f"Bearer {self._token()}", "Accept": accept or "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "discord-repo-bot"}
        resp = None
        for attempt in range(1, 4):
            try:
                resp = self.session.request(method, API + path, headers=headers, params=params, json=body,
                                            timeout=rc.HTTP_TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 3:
                    raise GitHubError(f"Couldn't reach GitHub ({type(e).__name__}).") from None
                self._sleep(1.5 * attempt)
                continue
            if resp.status_code in (502, 503, 504) and attempt < 3:
                self._sleep(1.5 * attempt)
                continue
            break
        if resp.status_code in ok:
            return resp.json() if resp.content else {}
        if resp.status_code == 404 and not_found_ok:
            return None
        raise self._error(resp, path)

    def _error(self, resp, path) -> GitHubError:
        try:
            data = resp.json()
            msg = str(data.get("message", "")) if isinstance(data, dict) else ""
        except ValueError:
            msg = ""
        msg = rc.redact(msg)[:300]
        code = resp.status_code
        hdr = getattr(resp, "headers", {}) or {}
        low = msg.lower()
        if code in (403, 429) and (hdr.get("X-RateLimit-Remaining") == "0" or "rate limit" in low or hdr.get("Retry-After")):
            wait = None
            try:
                if hdr.get("Retry-After"):
                    wait = float(hdr["Retry-After"])
                elif hdr.get("X-RateLimit-Reset"):
                    wait = max(0.0, float(hdr["X-RateLimit-Reset"]) - time.time())
            except ValueError:
                pass
            return GitHubRateLimit("GitHub rate limit reached." + (f" Try again in about {int(wait) + 1} s." if wait else ""),
                                   status=code, retry_after=wait)
        if code == 401:
            return GitHubAuthError("GitHub rejected the token (401). The owner must update GITHUB_TOKEN in botkey.py.",
                                   status=code)
        if code == 403:
            return GitHubAuthError(f"The GitHub token lacks permission for this action (403: {msg}).", status=code)
        if code == 404:
            return GitHubNotFound("GitHub says not found (404): the repo/branch/file doesn't exist, or the token has no "
                                  "access to it.", status=code)
        if code == 409:
            return GitHubConflict(f"GitHub conflict (409): {msg}", status=code)
        return GitHubError(f"GitHub error {code}: {msg or 'request failed'}", status=code)

    # -- reads -----------------------------------------------------------
    def get_repo(self, repo: str) -> dict:
        return self._request("GET", f"/repos/{repo}")

    def get_branch_sha(self, repo: str, branch: str) -> str | None:
        data = self._request("GET", f"/repos/{repo}/git/ref/heads/{_q(branch)}", not_found_ok=True)
        return data["object"]["sha"] if data else None

    def _commit_tree_sha(self, repo: str, commit_sha: str) -> str:
        return self._request("GET", f"/repos/{repo}/git/commits/{commit_sha}")["tree"]["sha"]

    def list_tree(self, repo: str, ref: str) -> TreeListing:
        """All files (blobs) at branch `ref`. Target repos only (must be on the allowlist)."""
        rc.policy_for(repo)
        sha = self.get_branch_sha(repo, ref)
        if sha is None:
            raise GitHubNotFound(f"Branch `{ref}` doesn't exist in `{repo}`.", status=404)
        data = self._request("GET", f"/repos/{repo}/git/trees/{sha}", params={"recursive": "1"})
        entries = [TreeEntry(t["path"], int(t.get("size") or 0), t.get("mode", "100644"), t["sha"])
                   for t in data.get("tree", []) if t.get("type") == "blob"]
        return TreeListing(sha, entries, bool(data.get("truncated")))

    def read_file(self, repo: str, path: str, ref: str) -> FileContent:
        """One file's text at branch `ref`. Binary files come back with is_binary=True and no text."""
        rc.policy_for(repo)
        path = rc.normalize_path(path)
        data = self._request("GET", f"/repos/{repo}/contents/{_q(path)}", params={"ref": ref})
        if isinstance(data, list):
            raise ValidationError(f"`{path}` is a directory, not a file.")
        if data.get("type") != "file":
            raise ValidationError(f"`{path}` is a {data.get('type', 'non-file')}, not a regular file.")
        size = int(data.get("size") or 0)
        if size > rc.MAX_FETCH_BYTES:
            raise LimitExceeded(f"`{path}` is too large ({_human(size)}); the bot won't read files over "
                                f"{rc.MAX_FETCH_BYTES // 1000} kB.")
        content = data.get("content") or ""
        if not content and size:                      # contents API omits bodies in some cases: use the blob API
            blob = self._request("GET", f"/repos/{repo}/git/blobs/{data['sha']}")
            content = blob.get("content", "")
        raw = base64.b64decode(content) if content else b""
        if b"\x00" in raw[:8192]:
            return FileContent(path, "", size, data.get("sha", ""), is_binary=True)
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return FileContent(path, "", size, data.get("sha", ""), is_binary=True)
        return FileContent(path, text.replace("\r\n", "\n"), size, data.get("sha", ""))

    def view(self, repo: str, ref: str) -> "RepoView":
        """Read-only handle bound to ONE repo + branch (what tools read through; the model can't change it)."""
        policy = rc.policy_for(repo)
        if ref not in policy.bases:
            rc.validate_head_branch(ref, policy)
        return RepoView(self, policy.name, ref)

    # -- low-level git data ---------------------------------------------
    def _create_blob(self, repo: str, content) -> str:
        if isinstance(content, bytes):
            body = {"content": base64.b64encode(content).decode("ascii"), "encoding": "base64"}
        else:
            body = {"content": content, "encoding": "utf-8"}
        return self._request("POST", f"/repos/{repo}/git/blobs", body=body)["sha"]

    def _build_entries(self, repo: str, changes: dict, modes: dict | None) -> list:
        entries = []
        for path, content in sorted(changes.items()):
            if content is None:
                entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            else:
                entries.append({"path": path, "mode": (modes or {}).get(path, "100644"), "type": "blob",
                                "sha": self._create_blob(repo, content)})
        return entries

    @staticmethod
    def _check_changes(changes: dict):
        if not changes:
            raise ValidationError("Nothing to commit.")
        if len(changes) > rc.MAX_FILES_PER_TASK:
            raise LimitExceeded(f"Too many files ({len(changes)}; max {rc.MAX_FILES_PER_TASK}).")
        total = 0
        for path, content in changes.items():
            rc.normalize_path(path, for_write=True)
            if content is None:
                continue
            text = content if isinstance(content, str) else content.decode("utf-8", "ignore")
            total += len(text)
            label = rc.find_secret(text)
            if label:
                raise SecretLeakError(f"Refusing to commit `{path}`: it contains {label}.")
        if total > rc.MAX_TOTAL_CHARS:
            raise LimitExceeded(f"Total size too large ({total} chars; max {rc.MAX_TOTAL_CHARS}).")

    @staticmethod
    def commit_url(repo: str, sha: str) -> str:
        return f"https://github.com/{repo}/commit/{sha}"

    # -- sandbox: orphan commit + force push (history intentionally lost) ----
    def sandbox_force_push(self, files: dict, message: str, branch: str | None = None) -> CommitResult:
        """Replace the whole sandbox branch with ONE parentless commit containing exactly `files`
        ({path: str | bytes}). Repo and default branch come from botkey, never from the caller/model."""
        repo = rc.require_sandbox_repo()
        branch = rc.check_branch_name(branch) if branch else rc.cfg().sandbox_branch
        self._check_changes({p: c for p, c in files.items() if c is not None})
        if any(c is None for c in files.values()):
            raise ValidationError("Sandbox pushes contain files only (no deletions).")
        message = rc.sanitize_commit_message(message, "Sandbox output")
        try:
            entries = self._build_entries(repo, files, None)
        except GitHubConflict as e:
            if "empty" not in str(e).lower():
                raise
            self._init_empty_repo(repo)
            entries = self._build_entries(repo, files, None)
        tree = self._request("POST", f"/repos/{repo}/git/trees", body={"tree": entries})["sha"]
        commit = self._request("POST", f"/repos/{repo}/git/commits",
                               body={"message": message, "tree": tree, "parents": []})["sha"]
        if self.get_branch_sha(repo, branch) is None:
            self._request("POST", f"/repos/{repo}/git/refs", body={"ref": f"refs/heads/{branch}", "sha": commit})
        else:
            self._request("PATCH", f"/repos/{repo}/git/refs/heads/{_q(branch)}", body={"sha": commit, "force": True})
        return CommitResult(commit, self.commit_url(repo, commit), branch)

    def _init_empty_repo(self, repo: str):
        """The Git Data API refuses to work on a repo with no commits: create one via the contents API."""
        if repo.lower() != rc.require_sandbox_repo().lower():
            raise RepoBotError("Refusing to initialise a non-sandbox repo.")
        self._request("PUT", f"/repos/{repo}/contents/README.md",
                      body={"message": "Initialise sandbox", "content": base64.b64encode(b"sandbox\n").decode("ascii")})

    # -- target: normal (non-force) commit on a bot branch ------------------
    def commit_to_branch(self, repo: str, branch: str, base: str, changes: dict, message: str,
                         modes: dict | None = None) -> CommitResult:
        """Commit `changes` ({path: text | None-for-delete}) on top of `branch`. If `branch` doesn't exist it is
        created from the tip of `base`. Deletes must refer to files that exist. Never force-pushes."""
        policy = rc.assert_pushable(repo, branch, base)
        repo = policy.name
        self._check_changes(changes)
        message = rc.sanitize_commit_message(message)
        branch_sha = self.get_branch_sha(repo, branch)
        parent = branch_sha or self.get_branch_sha(repo, base)
        if parent is None:
            raise GitHubNotFound(f"Base branch `{base}` doesn't exist in `{repo}`.", status=404)
        base_tree = self._commit_tree_sha(repo, parent)
        entries = self._build_entries(repo, changes, modes)
        tree = self._request("POST", f"/repos/{repo}/git/trees", body={"base_tree": base_tree, "tree": entries})["sha"]
        commit = self._request("POST", f"/repos/{repo}/git/commits",
                               body={"message": message, "tree": tree, "parents": [parent]})["sha"]
        if branch_sha is None:
            self._request("POST", f"/repos/{repo}/git/refs", body={"ref": f"refs/heads/{branch}", "sha": commit})
        else:
            self._request("PATCH", f"/repos/{repo}/git/refs/heads/{_q(branch)}", body={"sha": commit, "force": False})
        return CommitResult(commit, self.commit_url(repo, commit), branch)

    # -- pull requests (open / update only; merging does not exist here) ----
    def find_open_pr(self, repo: str, head_branch: str) -> PullRequest | None:
        policy = rc.policy_for(repo)
        owner = policy.name.split("/")[0]
        data = self._request("GET", f"/repos/{policy.name}/pulls",
                             params={"state": "open", "head": f"{owner}:{head_branch}", "per_page": 1})
        return self._pr(data[0]) if data else None

    def list_open_bot_prs(self, repo: str) -> list:
        """Open PRs from bot branches (prefix), most recently updated first."""
        policy = rc.policy_for(repo)
        prefix = rc.cfg().branch_prefix
        data = self._request("GET", f"/repos/{policy.name}/pulls",
                             params={"state": "open", "sort": "updated", "direction": "desc", "per_page": 50})
        return [self._pr(d) for d in data if str(d.get("head", {}).get("ref", "")).startswith(prefix)
                and (d.get("head", {}).get("repo") or {}).get("full_name", "").lower() == policy.name.lower()]

    def latest_bot_branch(self, repo: str) -> str | None:
        """Branch of the most recently updated open bot PR (what `/repo task iter:True` continues on)."""
        prs = self.list_open_bot_prs(repo)
        return prs[0].head if prs else None

    @staticmethod
    def _pr(d: dict, reused: bool = False) -> PullRequest:
        return PullRequest(int(d["number"]), d["html_url"], d["head"]["ref"], d["base"]["ref"], d.get("title", ""), reused)

    def open_pr(self, repo: str, head: str, base: str, title: str, body: str) -> PullRequest:
        policy = rc.assert_pushable(repo, head, base)
        repo = policy.name
        title, body = rc.sanitize_pr_text(title, body)
        existing = self.find_open_pr(repo, head)
        if existing:
            existing.reused = True
            return existing
        try:
            data = self._request("POST", f"/repos/{repo}/pulls",
                                 body={"title": title, "head": head, "base": base, "body": body,
                                       "maintainer_can_modify": False})
        except GitHubError as e:
            if e.status == 422 and "no commits between" in str(e).lower():
                raise GitHubError("Nothing to open a PR for: the branch has no changes compared with the base.",
                                  status=422) from None
            raise
        return self._pr(data)

    def update_pr(self, repo: str, number: int, title: str | None = None, body: str | None = None) -> PullRequest:
        policy = rc.policy_for(repo)
        payload = {}
        if title is not None or body is not None:
            t, b = rc.sanitize_pr_text(title, body)
            if title is not None:
                payload["title"] = t
            if body is not None:
                payload["body"] = b
        data = self._request("PATCH", f"/repos/{policy.name}/pulls/{int(number)}", body=payload)
        return self._pr(data)


# ---------------------------------------------------------------------------
# RepoView: read-only, bound to one repo + ref, lazily lists the tree once.
# aitools.ToolSession reads through this; nothing in it takes a repo/branch from the model.
# ---------------------------------------------------------------------------
class RepoView:
    def __init__(self, client: GitHubClient, repo: str, ref: str):
        self.client, self.repo, self.ref = client, repo, ref
        self._listing: TreeListing | None = None
        self._cache: dict = {}

    @property
    def listing(self) -> TreeListing:
        if self._listing is None:
            self._listing = self.client.list_tree(self.repo, self.ref)
        return self._listing

    def entries(self) -> list:
        return self.listing.entries

    @property
    def truncated(self) -> bool:
        return self.listing.truncated

    def read(self, path: str) -> FileContent:
        if path not in self._cache:
            self._cache[path] = self.client.read_file(self.repo, path, self.ref)
        return self._cache[path]
