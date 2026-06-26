"""Git context capture for agentic CLI metamodels."""
import difflib
import hashlib
import json
import logging
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_TIMEOUT = 2.0
_MAX_UNTRACKED_BYTES = 1_000_000
_MAX_SNAPSHOT_FILE_BYTES = 1_000_000
_MAX_SNAPSHOT_FILES = 5000
_SKIP_DIRS = {
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
}


def _git(*args, cwd: Optional[Path] = None):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL, text=True, timeout=_TIMEOUT,
        ).rstrip("\n")
    except Exception:
        return None


def _is_state_path(path):
    return path.startswith(".monocle/") or "/.monocle/" in path


def _is_hidden_path(path: Union[str, Path]) -> bool:
    return any(part.startswith(".") for part in Path(path).parts)


def _repo_name(url):
    if not url:
        return ""
    m = re.search(r"[/:]([^/:]+?)(?:\.git)?/?$", url)
    return m.group(1) if m else url


def _resolve_repo_root(cwd: Optional[Union[str, Path]] = None) -> Optional[Path]:
    candidate = Path(cwd).expanduser() if cwd else Path.cwd()
    if candidate.is_file():
        candidate = candidate.parent
    root = _git("rev-parse", "--show-toplevel", cwd=candidate)
    return Path(root) if root else None


def _resolve_workspace_root(cwd: Optional[Union[str, Path]] = None) -> Optional[Path]:
    repo_root = _resolve_repo_root(cwd)
    if repo_root:
        return repo_root
    candidate = Path(cwd).expanduser() if cwd else Path.cwd()
    if candidate.is_file():
        candidate = candidate.parent
    try:
        return candidate.resolve() if candidate.exists() else None
    except OSError:
        return None


def _snapshot(cwd: Optional[Union[str, Path]] = None):
    repo_root = _resolve_repo_root(cwd)
    if not repo_root:
        return {}
    head = _git("rev-parse", "HEAD", cwd=repo_root)
    branch = (
        _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root)
        or _git("symbolic-ref", "--short", "HEAD", cwd=repo_root)
        or ""
    ).strip()
    return {
        "head_sha": head.strip() if head else "",
        "branch": branch,
        "repo_url": (_git("remote", "get-url", "origin", cwd=repo_root) or "").strip(),
        "repo_root": str(repo_root),
        "is_submodule": bool((_git("rev-parse", "--show-superproject-working-tree", cwd=repo_root) or "").strip()),
    }


def _uncommitted_count(cwd: Optional[Path] = None):
    # Porcelain output is column-sensitive; bypass _git()'s rstrip via direct call.
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
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


def _commit_count_since(base_sha, cwd: Optional[Path] = None):
    out = _git("rev-list", "--count", f"{base_sha}..HEAD", cwd=cwd)
    try:
        return int(out) if out else 0
    except ValueError:
        return 0


def _diff_stats(base_sha, cwd: Optional[Path] = None):
    files, added, removed = [], 0, 0
    out = _git("diff", "--numstat", base_sha, cwd=cwd)
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


def _untracked_meta(cwd: Optional[Path] = None):
    meta = {}
    out = _git("ls-files", "--others", "--exclude-standard", cwd=cwd)
    if not out:
        return meta
    for p in out.splitlines():
        if not p or _is_state_path(p):
            continue
        try:
            st = (cwd / p if cwd else Path(p)).stat()
            meta[p] = [st.st_size, st.st_mtime]
        except OSError:
            meta[p] = [0, 0]
    return meta


def _untracked_changes(baseline_meta, cwd: Optional[Path] = None):
    current = _untracked_meta(cwd)
    changed = [p for p, meta in current.items() if baseline_meta.get(p) != meta]
    files, added = [], 0
    for path in sorted(changed):
        p = cwd / path if cwd else Path(path)
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


def _is_binary(data: bytes) -> bool:
    return b"\0" in data


def _text_lines(text: str) -> List[str]:
    return text.splitlines()


def _line_delta(before: str, after: str) -> Tuple[int, int]:
    added = removed = 0
    matcher = difflib.SequenceMatcher(a=_text_lines(before), b=_text_lines(after), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            removed += i2 - i1
            added += j2 - j1
    return added, removed


def _workspace_snapshot(root: Optional[Path]):
    if not root:
        return {}
    files = {}
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        base = Path(dirpath)
        for name in filenames:
            if name.startswith("."):
                continue
            path = base / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                if _is_state_path(rel) or _is_hidden_path(rel):
                    continue
                stat = path.stat()
                if stat.st_size > _MAX_SNAPSHOT_FILE_BYTES:
                    files[rel] = {"size": stat.st_size, "hash": "", "text": None}
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            scanned += 1
            if scanned > _MAX_SNAPSHOT_FILES:
                files["__snapshot_truncated__"] = {"size": 0, "hash": "", "text": None}
                return files
            digest = hashlib.sha256(data).hexdigest()
            if _is_binary(data):
                files[rel] = {"size": stat.st_size, "hash": digest, "text": None}
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="ignore")
            files[rel] = {"size": stat.st_size, "hash": digest, "text": text}
    return files


def _workspace_changes(baseline_files, current_files):
    paths = sorted((set(baseline_files) | set(current_files)) - {"__snapshot_truncated__"})
    changed, added, removed = [], 0, 0
    for path in paths:
        before = baseline_files.get(path)
        after = current_files.get(path)
        if before == after:
            continue
        if before and after and before.get("hash") == after.get("hash"):
            continue
        changed.append(path)
        before_text = before.get("text") if before else ""
        after_text = after.get("text") if after else ""
        if before_text is None or after_text is None:
            continue
        if before is None:
            added += len(_text_lines(after_text))
        elif after is None:
            removed += len(_text_lines(before_text))
        else:
            a, r = _line_delta(before_text, after_text)
            added += a
            removed += r
    return {"files": changed, "added": added, "removed": removed}


def apply_to_span(span, kwargs):
    """Set scope.* attributes directly on the turn span only.
    Call from pre_task_processing on the handle_turn wrap site."""
    for key, value in (kwargs.get("git_scopes") or {}).items():
        span.set_attribute(f"scope.{key}", value)


class GitContext:
    def __init__(self, sessions_dir_fn: Callable[[], Path], file_prefix: str):
        self._sessions_dir = sessions_dir_fn
        self._prefix = file_prefix

    def _baseline_file(self, session_id):
        return self._sessions_dir() / f"{self._prefix}_{session_id}.turn_baseline.json"

    def capture_turn_baseline(self, session_id, cwd: Optional[Union[str, Path]] = None):
        snap = _snapshot(cwd)
        workspace_root = _resolve_workspace_root(cwd)
        if not snap and not workspace_root:
            return
        root = Path(snap.get("repo_root")) if snap and snap.get("repo_root") else workspace_root
        f = self._baseline_file(session_id)
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({
                "head_sha": snap.get("head_sha", "") if snap else "",
                "branch": snap.get("branch", "") if snap else "",
                "repo_root": snap.get("repo_root", "") if snap else "",
                "workspace_root": str(root) if root else "",
                "files": _workspace_snapshot(root),
                "untracked": _untracked_meta(Path(snap["repo_root"])) if snap and snap.get("repo_root") else {},
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

    def compute_scopes(self, session_id, include_turn_deltas=True, cwd: Optional[Union[str, Path]] = None):
        """Build scope.git.* attributes for the current working tree.

        Snapshot fields (repo, branch, commit, uncommitted) are always
        included. Per-turn deltas (edit.turn.*) are diffed against the
        baseline captured at the start of the turn — set include_turn_deltas=False
        to skip them. Codex does this because its replay can lag a turn behind the
        working tree (transcript-flush race), so wall-clock git deltas would be
        misattributed; it derives edit.turn.* from the transcript patches instead.
        """
        baseline = self._load_baseline(session_id)
        repo_cwd = cwd or baseline.get("repo_root") or baseline.get("workspace_root")
        current = _snapshot(repo_cwd)
        workspace_root = _resolve_workspace_root(repo_cwd)
        if current:
            repo_root = Path(current["repo_root"])
        elif workspace_root:
            repo_root = workspace_root
        else:
            return {"git.status": "no repo connected"}

        scopes = {}
        if current:
            scopes.update({
                "git.repo": _repo_name(current["repo_url"]),
                "git.repo.url": current["repo_url"],
                "git.branch": current["branch"],
                "git.commit.hash": current["head_sha"],
                "git.uncommitted": _uncommitted_count(repo_root),
            })
            if current["is_submodule"]:
                scopes["git.is_submodule"] = True
        else:
            scopes["git.status"] = "no repo connected"

        if include_turn_deltas:
            if not baseline:
                # Don't write current state as the baseline file — a subsequent
                # compute_scopes call in the same Stop would then see ZERO delta.
                # Just use an empty in-memory baseline; deltas degrade to 0.
                baseline = {
                    "head_sha": current.get("head_sha", "") if current else "",
                    "branch": current.get("branch", "") if current else "",
                    "files": _workspace_snapshot(repo_root),
                    "untracked": {},
                }

            scopes["edit.turn.files_changed"] = 0
            scopes["edit.turn.lines_added"] = 0
            scopes["edit.turn.lines_removed"] = 0
            scopes["edit.turn.commits_added"] = 0
            scopes["edit.turn.file_types"] = ""
            if baseline.get("branch") and baseline["branch"] != current["branch"]:
                scopes["edit.turn.branch_changed"] = True

            base_sha = baseline.get("head_sha", "")
            if baseline.get("files"):
                diff = _workspace_changes(baseline.get("files", {}), _workspace_snapshot(repo_root))
            else:
                diff = {"files": [], "added": 0, "removed": 0}
                if base_sha:
                    diff = _diff_stats(base_sha, repo_root)
                untracked = _untracked_changes(baseline.get("untracked", {}), repo_root)
                diff["files"] += untracked["files"]
                diff["added"] += untracked["added"]
            if base_sha and current:
                scopes["edit.turn.commits_added"] = _commit_count_since(base_sha, repo_root)
            paths = diff["files"]
            scopes["edit.turn.files_changed"] = len(paths)
            scopes["edit.turn.lines_added"] = diff["added"]
            scopes["edit.turn.lines_removed"] = diff["removed"]
            scopes["edit.turn.file_types"] = _file_types_summary(paths)

        return {k: v for k, v in scopes.items() if v not in ("", None)}

    def cleanup(self, session_id):
        self._baseline_file(session_id).unlink(missing_ok=True)
