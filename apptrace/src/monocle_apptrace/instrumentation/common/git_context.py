"""Git context capture for agentic CLI metamodels."""
import json
import logging
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_TIMEOUT = 2.0
_MAX_UNTRACKED_BYTES = 1_000_000


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL, text=True, timeout=_TIMEOUT,
        ).rstrip("\n")
    except Exception:
        return None


def _is_state_path(path):
    return path.startswith(".monocle/") or "/.monocle/" in path


def _repo_name(url):
    if not url:
        return ""
    m = re.search(r"[/:]([^/:]+?)(?:\.git)?/?$", url)
    return m.group(1) if m else url


def _snapshot():
    head = _git("rev-parse", "HEAD")
    if not head:
        return {}
    return {
        "head_sha": head.strip(),
        "branch": (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip(),
        "repo_url": (_git("remote", "get-url", "origin") or "").strip(),
        "is_submodule": bool((_git("rev-parse", "--show-superproject-working-tree") or "").strip()),
    }


def _uncommitted_count():
    # Porcelain output is column-sensitive; bypass _git()'s rstrip via direct call.
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True, timeout=_TIMEOUT,
        )
    except Exception:
        return 0
    n = 0
    for line in out.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].split(" -> ")[-1]
        if not _is_state_path(path):
            n += 1
    return n


def _commit_count_since(base_sha):
    out = _git("rev-list", "--count", f"{base_sha}..HEAD")
    try:
        return int(out) if out else 0
    except ValueError:
        return 0


def _diff_stats(base_sha):
    files, added, removed = [], 0, 0
    out = _git("diff", "--numstat", base_sha)
    if not out:
        return {"files": files, "added": added, "removed": removed}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        path = parts[2]
        if _is_state_path(path):
            continue
        files.append(path)
        if parts[0] != "-":
            try: added += int(parts[0])
            except ValueError: pass
        if parts[1] != "-":
            try: removed += int(parts[1])
            except ValueError: pass
    return {"files": files, "added": added, "removed": removed}


def _untracked_meta():
    meta = {}
    out = _git("ls-files", "--others", "--exclude-standard")
    if not out:
        return meta
    for p in out.splitlines():
        if not p or _is_state_path(p):
            continue
        try:
            st = Path(p).stat()
            meta[p] = [st.st_size, st.st_mtime]
        except OSError:
            meta[p] = [0, 0]
    return meta


def _untracked_changes(baseline_meta):
    current = _untracked_meta()
    changed = [p for p, meta in current.items() if baseline_meta.get(p) != meta]
    files, added = [], 0
    for path in sorted(changed):
        p = Path(path)
        try:
            if p.stat().st_size > _MAX_UNTRACKED_BYTES:
                files.append(path)
                continue
            content = p.read_text(encoding="utf-8", errors="ignore")
        except (OSError, IsADirectoryError):
            files.append(path)
            continue
        n = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        files.append(path)
        added += n
    return {"files": files, "added": added}


def _file_types_summary(paths):
    if not paths:
        return ""
    counts = Counter(Path(p).suffix or "(none)" for p in paths)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ",".join(f"{ext}:{n}" for ext, n in items)


def apply_to_span(span, kwargs):
    """Set scope.git.* attributes directly on the turn span only.
    Call from pre_task_processing on the handle_turn wrap site."""
    for key, value in (kwargs.get("git_scopes") or {}).items():
        span.set_attribute(f"scope.{key}", value)


class GitContext:
    def __init__(self, sessions_dir_fn: Callable[[], Path], file_prefix: str):
        self._sessions_dir = sessions_dir_fn
        self._prefix = file_prefix

    def _baseline_file(self, session_id):
        return self._sessions_dir() / f"{self._prefix}_{session_id}.turn_baseline.json"

    def capture_turn_baseline(self, session_id):
        snap = _snapshot()
        if not snap:
            return
        f = self._baseline_file(session_id)
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({
                "head_sha": snap["head_sha"],
                "branch": snap["branch"],
                "untracked": _untracked_meta(),
            }))
        except Exception as e:
            logger.debug("turn baseline capture failed: %s", e)

    def _load_baseline(self, session_id):
        f = self._baseline_file(session_id)
        if not f.exists():
            return {}
        try:
            data = json.loads(f.read_text())
        except Exception:
            return {}
        if not isinstance(data.get("untracked", {}), dict):
            return {}
        return data

    def compute_scopes(self, session_id, include_turn_deltas=True):
        """Build scope.git.* attributes for the current working tree.

        Snapshot fields (repo, branch, commit, uncommitted) are always
        included. Per-turn deltas (git.turn.*) are diffed against the
        baseline captured at the start of the turn — set include_turn_deltas=False
        to skip them. Codex does this because its replay can lag a turn behind the
        working tree (transcript-flush race), so wall-clock git deltas would be
        misattributed; it derives edit.turn.* from the transcript patches instead.
        """
        current = _snapshot()
        if not current:
            return {"git.status": "no repo connected"}

        scopes = {
            "git.repo": _repo_name(current["repo_url"]),
            "git.repo.url": current["repo_url"],
            "git.branch": current["branch"],
            "git.commit.hash": current["head_sha"],
            "git.uncommitted": _uncommitted_count(),
        }
        if current["is_submodule"]:
            scopes["git.is_submodule"] = True

        if include_turn_deltas:
            baseline = self._load_baseline(session_id)
            if not baseline:
                # Don't write current state as the baseline file — a subsequent
                # compute_scopes call in the same Stop would then see ZERO delta.
                # Just use an empty in-memory baseline; deltas degrade to 0.
                baseline = {"head_sha": current["head_sha"], "branch": current["branch"], "untracked": {}}

            scopes["git.turn.files_changed"] = 0
            scopes["git.turn.lines_added"] = 0
            scopes["git.turn.lines_removed"] = 0
            scopes["git.turn.commits_added"] = 0
            scopes["git.turn.file_types"] = ""
            if baseline.get("branch") and baseline["branch"] != current["branch"]:
                scopes["git.turn.branch_changed"] = True

            base_sha = baseline.get("head_sha", "")
            if base_sha:
                diff = _diff_stats(base_sha)
                untracked = _untracked_changes(baseline.get("untracked", {}))
                paths = diff["files"] + untracked["files"]
                scopes["git.turn.files_changed"] = len(paths)
                scopes["git.turn.lines_added"] = diff["added"] + untracked["added"]
                scopes["git.turn.lines_removed"] = diff["removed"]
                scopes["git.turn.commits_added"] = _commit_count_since(base_sha)
                scopes["git.turn.file_types"] = _file_types_summary(paths)

        return {k: v for k, v in scopes.items() if v not in ("", None)}

    def cleanup(self, session_id):
        self._baseline_file(session_id).unlink(missing_ok=True)
