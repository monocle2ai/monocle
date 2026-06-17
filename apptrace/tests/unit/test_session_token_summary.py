"""Unit tests for monocle_apptrace.session_token_summary"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from monocle_apptrace.session_token_summary import format_table, summarize

SESSION_ATTR = "scope.agentic.session"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span(session_id, model, prompt, completion, total, cache_read=0, cache_create=0):
    return {
        "attributes": {
            SESSION_ATTR: session_id,
            "entity.2.name": model,
        },
        "events": [
            {"name": "data.input", "attributes": {}},
            {"name": "data.output", "attributes": {}},
            {
                "name": "metadata",
                "attributes": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": cache_create,
                },
            },
        ],
    }


def _write_trace(directory, timestamp, spans):
    fname = "monocle_trace_test_workflow_{}.json".format(timestamp)
    (Path(directory) / fname).write_text(json.dumps(spans), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests: summarize()
# ---------------------------------------------------------------------------

class TestSummarize(unittest.TestCase):

    def test_missing_directory_returns_empty(self):
        rows = summarize("all", monocle_dir=Path("/tmp/nonexistent_monocle_xyz"))
        self.assertEqual(rows, [])

    def test_empty_directory_returns_empty(self):
        with TemporaryDirectory() as tmp:
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_single_span_single_session(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-30_10.00.00", [
                _span("session-abc", "gpt-4o", 100, 50, 150)
            ])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["session"], "session-abc")
            self.assertEqual(rows[0]["model"], "gpt-4o")
            self.assertEqual(rows[0]["prompt_tokens"], 100)
            self.assertEqual(rows[0]["completion_tokens"], 50)
            self.assertEqual(rows[0]["total_tokens"], 150)

    def test_multiple_spans_same_session_aggregated(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-30_10.00.00", [
                _span("session-abc", "gpt-4o", 100, 50, 150),
                _span("session-abc", "gpt-4o", 200, 80, 280),
            ])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt_tokens"], 300)
            self.assertEqual(rows[0]["completion_tokens"], 130)
            self.assertEqual(rows[0]["total_tokens"], 430)

    def test_different_sessions_not_aggregated(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-30_10.00.00", [
                _span("session-abc", "gpt-4o", 100, 50, 150),
                _span("session-xyz", "gpt-4o", 200, 80, 280),
            ])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 2)
            sessions = {r["session"] for r in rows}
            self.assertIn("session-abc", sessions)
            self.assertIn("session-xyz", sessions)

    def test_same_session_different_models(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-30_10.00.00", [
                _span("session-abc", "gpt-4o", 100, 50, 150),
                _span("session-abc", "claude-3-5-sonnet", 200, 80, 280),
            ])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["session"], "session-abc")
            models = {r["model"] for r in rows}
            self.assertIn("gpt-4o", models)
            self.assertIn("claude-3-5-sonnet", models)

    def test_cache_tokens_extracted(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-30_10.00.00", [
                _span("session-abc", "claude-3-5-sonnet", 100, 50, 150,
                      cache_read=40, cache_create=10)
            ])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows[0]["cache_read_input_tokens"], 40)
            self.assertEqual(rows[0]["cache_creation_input_tokens"], 10)

    def test_span_without_session_skipped(self):
        with TemporaryDirectory() as tmp:
            span_no_session = {
                "attributes": {"entity.2.name": "gpt-4o"},
                "events": [{"name": "metadata", "attributes": {
                    "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150
                }}],
            }
            _write_trace(tmp, "2025-11-30_10.00.00", [span_no_session])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_span_without_model_skipped(self):
        with TemporaryDirectory() as tmp:
            span_no_model = {
                "attributes": {SESSION_ATTR: "session-abc"},
                "events": [{"name": "metadata", "attributes": {
                    "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150
                }}],
            }
            _write_trace(tmp, "2025-11-30_10.00.00", [span_no_model])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_span_without_metadata_event_skipped(self):
        with TemporaryDirectory() as tmp:
            span_no_meta = {
                "attributes": {SESSION_ATTR: "session-abc", "entity.2.name": "gpt-4o"},
                "events": [{"name": "data.input", "attributes": {}}],
            }
            _write_trace(tmp, "2025-11-30_10.00.00", [span_no_meta])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_malformed_json_file_skipped(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "monocle_trace_bad_2025-11-30_10.00.00.json").write_text(
                "NOT JSON", encoding="utf-8"
            )
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_session_spans_across_multiple_files(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-29_10.00.00", [
                _span("session-abc", "gpt-4o", 100, 50, 150)
            ])
            _write_trace(tmp, "2025-11-30_10.00.00", [
                _span("session-abc", "gpt-4o", 200, 80, 280)
            ])
            rows = summarize("all", monocle_dir=Path(tmp))
            # same session across files should aggregate
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt_tokens"], 300)
            self.assertEqual(rows[0]["total_tokens"], 430)


# ---------------------------------------------------------------------------
# Tests: time window filtering
# ---------------------------------------------------------------------------

class TestTimeWindowFiltering(unittest.TestCase):

    def test_today_excludes_old_files(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2020-01-01_10.00.00", [
                _span("session-abc", "gpt-4o", 100, 50, 150)
            ])
            rows = summarize("today", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_7_days_excludes_old_files(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2020-01-01_10.00.00", [
                _span("session-abc", "gpt-4o", 100, 50, 150)
            ])
            rows = summarize("7 days", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_all_includes_old_files(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2020-01-01_10.00.00", [
                _span("session-abc", "gpt-4o", 100, 50, 150)
            ])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 1)

    def test_unknown_window_treated_as_all(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2020-01-01_10.00.00", [
                _span("session-abc", "gpt-4o", 100, 50, 150)
            ])
            rows = summarize("last quarter", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 1)


# ---------------------------------------------------------------------------
# Tests: format_table()
# ---------------------------------------------------------------------------

class TestFormatTable(unittest.TestCase):

    def test_empty_rows_message(self):
        out = format_table([])
        self.assertIn("No trace files found", out)

    def test_headers_present(self):
        out = format_table([_row()])
        for header in ["Session", "Model", "Input", "Cache Read", "Cache Create", "Output", "Total"]:
            self.assertIn(header, out)

    def test_data_present(self):
        out = format_table([_row()])
        self.assertIn("session-abc", out)
        self.assertIn("gpt-4o", out)
        self.assertIn("100", out)

    def test_multiple_rows(self):
        out = format_table([_row("session-abc"), _row("session-xyz", "claude-3-5-sonnet")])
        self.assertIn("session-abc", out)
        self.assertIn("session-xyz", out)

    def test_table_borders_present(self):
        out = format_table([_row()])
        self.assertIn("+", out)
        self.assertIn("|", out)
        self.assertIn("-", out)


def _row(session="session-abc", model="gpt-4o"):
    return {
        "session": session,
        "model": model,
        "prompt_tokens": 100,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


if __name__ == "__main__":
    unittest.main()