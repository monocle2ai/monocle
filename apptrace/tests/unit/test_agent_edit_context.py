"""Unit tests for GitContext / agent_edit_context.py (issue #690).

Tests verify that pre-existing dirty tracked files are NOT counted as
turn-delta changes when no per-file snapshot baseline is available.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FILE_PREFIX = "test"


def _make_baseline(head_sha="abc123", branch="main", files=None, untracked=None):
    """Return a baseline dict matching the structure _load_baseline() returns."""
    return {
        "head_sha": head_sha,
        "branch": branch,
        "files": files or {},     # empty = no snapshot baseline (triggers else-branch)
        "untracked": untracked or {},
    }


def _write_baseline(session_id, data, baseline_dir):
    """Write a baseline JSON file in *baseline_dir* using the same naming convention as GitContext."""
    p = Path(baseline_dir) / f"{_FILE_PREFIX}_{session_id}.turn_baseline.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeScopesNoOverReport(unittest.TestCase):
    """Issue #690: else-branch must only count committed changes, not dirty files."""

    def _make_context(self, baseline_dir):
        from monocle_apptrace.instrumentation.common.agent_edit_context import GitContext
        bdir = Path(baseline_dir)
        return GitContext(sessions_dir_fn=lambda: bdir, file_prefix=_FILE_PREFIX)

    def test_no_pre_existing_dirty_files_counted(self):
        """Pre-existing uncommitted tracked edits must NOT appear in edit.turn.files_changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "sess-001"
            # Baseline with no per-file snapshot so we hit the else-branch.
            baseline = _make_baseline(head_sha="deadbeef", files={})
            _write_baseline(session_id, baseline, tmpdir)

            ctx = self._make_context(tmpdir)

            # _snapshot returns current repo state
            fake_snapshot = {
                "repo_root": tmpdir,
                "repo_url": "https://github.com/test/repo.git",
                "branch": "main",
                "head_sha": "deadbeef",   # same SHA — no new commits
                "is_submodule": False,
            }

            # _diff_stats should be called with "deadbeef..HEAD" (two-dot range)
            # to only count committed changes. We return an empty diff to simulate
            # no new commits (same SHA → nothing committed since baseline).
            empty_diff = {"files": [], "added": 0, "removed": 0}

            with patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._snapshot",
                return_value=fake_snapshot,
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._resolve_workspace_root",
                return_value=Path(tmpdir),
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._diff_stats",
                return_value=empty_diff,
            ) as mock_diff, patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._uncommitted_count",
                return_value=3,   # 3 dirty tracked files pre-existed
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._untracked_changes",
                return_value={"files": [], "added": 0},
            ):
                scopes = ctx.compute_scopes(session_id)

            # The key fix: _diff_stats called with two-dot range, not bare SHA
            mock_diff.assert_called_once()
            call_args = mock_diff.call_args[0]
            self.assertIn("..", call_args[0],
                          "_diff_stats must use 'sha..HEAD' range, not bare SHA")

            # No files changed (no new commits, no untracked changes)
            self.assertEqual(scopes.get("edit.turn.files_changed", 0), 0,
                             "Pre-existing dirty files must not be counted")

    def test_committed_changes_are_counted(self):
        """Newly committed files after baseline ARE included in edit.turn.files_changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "sess-002"
            baseline = _make_baseline(head_sha="base001", files={})
            _write_baseline(session_id, baseline, tmpdir)

            ctx = self._make_context(tmpdir)

            fake_snapshot = {
                "repo_root": tmpdir,
                "repo_url": "https://github.com/test/repo.git",
                "branch": "main",
                "head_sha": "newsha1",
                "is_submodule": False,
            }

            committed_diff = {
                "files": ["src/new_feature.py", "tests/test_new_feature.py"],
                "added": 50,
                "removed": 5,
            }

            with patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._snapshot",
                return_value=fake_snapshot,
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._resolve_workspace_root",
                return_value=Path(tmpdir),
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._diff_stats",
                return_value=committed_diff,
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._uncommitted_count",
                return_value=0,
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._untracked_changes",
                return_value={"files": [], "added": 0},
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._commit_count_since",
                return_value=1,
            ):
                scopes = ctx.compute_scopes(session_id)

            self.assertEqual(scopes.get("edit.turn.files_changed", 0), 2)
            self.assertEqual(scopes.get("edit.turn.lines_added", 0), 50)
            self.assertEqual(scopes.get("edit.turn.lines_removed", 0), 5)

    def test_snapshot_baseline_path_unaffected(self):
        """When a per-file snapshot exists, the original _workspace_changes path is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "sess-003"
            # Non-empty files dict → snapshot path (not the else-branch we fixed)
            baseline = _make_baseline(head_sha="base001", files={"src/foo.py": [100, 99999]})
            _write_baseline(session_id, baseline, tmpdir)

            ctx = self._make_context(tmpdir)

            fake_snapshot = {
                "repo_root": tmpdir,
                "repo_url": "https://github.com/test/repo.git",
                "branch": "main",
                "head_sha": "base001",
                "is_submodule": False,
            }

            workspace_diff = {"files": ["src/foo.py"], "added": 10, "removed": 2}

            with patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._snapshot",
                return_value=fake_snapshot,
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._resolve_workspace_root",
                return_value=Path(tmpdir),
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._workspace_changes",
                return_value=workspace_diff,
            ) as mock_ws, patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._workspace_snapshot",
                return_value={"src/foo.py": [110, 99999]},
            ), patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._diff_stats",
            ) as mock_diff, patch(
                "monocle_apptrace.instrumentation.common.agent_edit_context._uncommitted_count",
                return_value=0,
            ):
                scopes = ctx.compute_scopes(session_id)

            # Snapshot path used _workspace_changes, not _diff_stats
            mock_ws.assert_called_once()
            mock_diff.assert_not_called()
            self.assertEqual(scopes.get("edit.turn.files_changed", 0), 1)


if __name__ == "__main__":
    unittest.main()
