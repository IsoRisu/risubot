# workspace.py
# The deterministic half of the Gemini-only repo workflow: no model, no network, standard library only.
#
#   ws = Workspace({"src/a.py": "x = 1\n"})                # the files the editor was shown (text as read from the repo)
#   reply = parse_edit_reply(model_text)                   # -> EditReply(summary, edits, cont); raises EditParseError
#   result = apply_edits(ws, reply.edits)                  # -> ApplyResult(applied, failed); never raises on a bad edit
#   problems = verify(ws, sorted(ws.changes()))            # -> [(path, error)]  (.py must compile, .json must parse)
#   ws.changes(restore_eol=True)                           # -> {path: new text | None (delete)}, only paths that differ
#
# Edit format (what the model sends, see repo_pipeline.EDIT_SYSTEM):
#   {"path", "op": "replace", "search": "<text that occurs exactly once>", "replace": "<new text>"}
#   {"path", "op": "create",  "replace": "<full content>"}        only for a path that does not exist (yet)
#   {"path", "op": "delete"}
#
# Rules and decisions:
#   * Edits are applied in order, each against the result of the previous ones. A failing edit changes nothing and is
#     recorded as (edit, reason); the other edits still apply ("partial failure keeps good edits").
#   * The workspace holds text with LF endings. Files that were CRLF are remembered in .normalized and get CRLF back in
#     changes(restore_eol=True). Search/replace text from the model is normalized to LF too.
#   * Failure reasons starting with LIMIT_PREFIX are size/count limit hits (the pipeline does not try to repair those).
#     Limits come from repoconfig: MAX_FILE_CHARS, MAX_FILES_PER_TASK, MAX_TOTAL_CHARS.
#   * Paths are validated with repoconfig.normalize_path(for_write=True) when the reply is parsed, so a bad path
#     makes the whole reply malformed (the model gets one retry with the message "edits[i].path: ...").
#   * Secret scanning is NOT done here; githubclient.commit_to_branch / sandbox_force_push refuse secrets.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import repoconfig as rc

LIMIT_PREFIX = "limit: "
MAX_EDITS_PER_REPLY = 100
OPS = ("replace", "create", "delete")


class EditParseError(rc.ValidationError):
    """The model's edit reply is malformed (the message is fed back to the model for the one retry)."""


@dataclass
class Edit:
    path: str
    op: str
    search: str | None = None
    replace: str | None = None


@dataclass
class EditReply:
    summary: str
    edits: list
    cont: bool = False


@dataclass
class ApplyResult:
    applied: list = field(default_factory=list)          # [Edit]
    failed: list = field(default_factory=list)           # [(Edit, reason)]


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.S)


def _lf(text: str) -> str:
    return text.replace("\r\n", "\n")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_edit_reply(text: str) -> EditReply:
    """Strict parse of an edit / repair reply. Raises EditParseError with a message the model can act on."""
    if not isinstance(text, str) or not text.strip():
        raise EditParseError("The reply was empty; expected a JSON object.")
    m = _FENCE_RE.match(text)
    if m:                                               # tolerate a markdown fence around the JSON
        text = m.group(1)
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as e:
        raise EditParseError(f"The reply is not valid JSON: {str(e)[:200]}")
    if not isinstance(data, dict):
        raise EditParseError("The reply must be a JSON object with \"summary\", \"edits\" and \"continue\".")
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        raise EditParseError("\"summary\" must be a string.")
    raw = data.get("edits")
    if not isinstance(raw, list):
        raise EditParseError("\"edits\" must be a list (use [] for no edits).")
    if len(raw) > MAX_EDITS_PER_REPLY:
        raise EditParseError(f"\"edits\" has {len(raw)} entries; at most {MAX_EDITS_PER_REPLY} per reply. "
                             "Send fewer edits and set \"continue\": true.")
    cont = data.get("continue", False)
    if not isinstance(cont, bool):
        raise EditParseError("\"continue\" must be true or false.")
    edits = []
    for i, item in enumerate(raw):
        where = f"edits[{i}]"
        if not isinstance(item, dict):
            raise EditParseError(f"{where} must be an object.")
        op = item.get("op")
        if op not in OPS:
            raise EditParseError(f"{where}.op must be one of \"replace\", \"create\", \"delete\".")
        try:
            path = rc.normalize_path(item.get("path"), for_write=True)
        except rc.ValidationError as e:
            raise EditParseError(f"{where}.path: {e.user_message}")
        search = replace = None
        if op == "replace":
            search, replace = item.get("search"), item.get("replace")
            if not isinstance(search, str) or not search:
                raise EditParseError(f"{where}.search must be a non-empty string for op \"replace\".")
            if not isinstance(replace, str):
                raise EditParseError(f"{where}.replace must be a string for op \"replace\".")
            search, replace = _lf(search), _lf(replace)
        elif op == "create":
            replace = item.get("replace")
            if not isinstance(replace, str):
                raise EditParseError(f"{where}.replace must hold the full file content (a string) for op \"create\".")
            replace = _lf(replace)
        edits.append(Edit(path, op, search, replace))
    return EditReply(summary.strip(), edits, cont)


# ---------------------------------------------------------------------------
# The in-memory workspace
# ---------------------------------------------------------------------------
class Workspace:
    def __init__(self, files: dict | None = None):
        self._orig: dict = {}               # path -> LF text as first seen
        self._cur: dict = {}                # path -> current LF text (absent = does not exist)
        self.normalized: set = set()        # paths that were CRLF when loaded
        self._touched: set = set()
        for path, text in (files or {}).items():
            if "\r\n" in text:
                self.normalized.add(path)
            self._orig[path] = self._cur[path] = _lf(text)

    def get(self, path: str):
        return self._cur.get(path)

    def exists(self, path: str) -> bool:
        return path in self._cur

    def touched(self) -> set:
        """Paths an applied edit has created, changed or deleted so far."""
        return set(self._touched)

    def _set(self, path: str, text) -> None:
        if text is None:
            self._cur.pop(path, None)
        else:
            self._cur[path] = text
        self._touched.add(path)

    def changes(self, restore_eol: bool = False) -> dict:
        """{path: new text | None}, only for paths whose final state differs from the start."""
        out = {}
        for path in sorted(self._touched):
            new, old = self._cur.get(path), self._orig.get(path)
            if new == old:
                continue
            if restore_eol and new is not None and path in self.normalized:
                new = new.replace("\n", "\r\n")
            out[path] = new
        return out

    def total_chars(self) -> int:
        return sum(len(t) for p, t in self._cur.items() if p in self._touched)


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------
def _apply_one(ws: Workspace, e: Edit):
    """-> None when applied, else the failure reason."""
    cur = ws.get(e.path)
    if e.op == "delete":
        if cur is None:
            return "file does not exist"
        ws._set(e.path, None)
        return None
    if e.op == "create":
        if cur is not None:
            return "file already exists (to replace a whole file send a delete edit and then a create edit)"
        new = e.replace or ""
    else:                                              # replace
        if cur is None:
            return "file does not exist"
        n = cur.count(e.search)
        if n == 0:
            return "search text not found in the file (it must match the current content exactly)"
        if n > 1:
            return f"search text found {n} times (it must occur exactly once; add surrounding lines)"
        new = cur.replace(e.search, e.replace, 1)
    if len(new) > rc.MAX_FILE_CHARS:
        return f"{LIMIT_PREFIX}the file would be {len(new)} chars, over the {rc.MAX_FILE_CHARS}-char file limit"
    if e.path not in ws._touched and len(ws._touched) >= rc.MAX_FILES_PER_TASK:
        return f"{LIMIT_PREFIX}more than {rc.MAX_FILES_PER_TASK} files changed in one task"
    total = ws.total_chars() - len(cur or "") * (e.path in ws._touched) + len(new)
    if total > rc.MAX_TOTAL_CHARS:
        return f"{LIMIT_PREFIX}the changed files would total over {rc.MAX_TOTAL_CHARS} chars"
    ws._set(e.path, new)
    return None


def apply_edits(ws: Workspace, edits) -> ApplyResult:
    """Apply in order. A failed edit leaves the workspace untouched and is reported; the rest still apply."""
    res = ApplyResult()
    for e in edits:
        why = _apply_one(ws, e)
        if why is None:
            res.applied.append(e)
        else:
            res.failed.append((e, why))
    return res


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(ws: Workspace, paths) -> list:
    """[(path, error)] for the given paths: .py must compile, .json must parse. Deleted files are skipped."""
    problems = []
    for p in paths:
        text = ws.get(p)
        if text is None:
            continue
        low = p.lower()
        if low.endswith(".py"):
            try:
                compile(text, p, "exec")
            except SyntaxError as e:
                problems.append((p, f"SyntaxError: {e.msg} (line {e.lineno})"))
            except (ValueError, RecursionError, MemoryError) as e:
                problems.append((p, f"SyntaxError: {str(e)[:150]}"))
        elif low.endswith(".json"):
            try:
                json.loads(text)
            except (ValueError, RecursionError) as e:
                problems.append((p, f"JSON error: {str(e)[:150]}"))
    return problems
