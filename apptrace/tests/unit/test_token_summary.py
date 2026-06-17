"""Unit tests for monocle_apptrace.token_summary"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from monocle_apptrace.token_summary import format_table, summarize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span(model, prompt, completion, total, cache_read=0, cache_create=0):
    """Build a minimal span dict with a metadata event."""
    return {
        "attributes": {"entity.2.name": model},
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
    """Write a trace JSON file. timestamp e.g. '2025-11-30_19.47.49'"""
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

    def test_single_span_single_model(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-30_10.00.00", [_span("gpt-4o", 100, 50, 150)])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], "gpt-4o")
            self.assertEqual(rows[0]["prompt_tokens"], 100)
            self.assertEqual(rows[0]["completion_tokens"], 50)
            self.assertEqual(rows[0]["total_tokens"], 150)

    def test_multiple_spans_same_model_aggregated(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-30_10.00.00", [
                _span("gpt-4o", 100, 50, 150),
                _span("gpt-4o", 200, 80, 280),
            ])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt_tokens"], 300)
            self.assertEqual(rows[0]["completion_tokens"], 130)
            self.assertEqual(rows[0]["total_tokens"], 430)

    def test_multiple_models_same_day(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-30_10.00.00", [
                _span("gpt-4o", 100, 50, 150),
                _span("claude-3-5-sonnet", 200, 80, 280),
            ])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 2)
            models = {r["model"] for r in rows}
            self.assertIn("gpt-4o", models)
            self.assertIn("claude-3-5-sonnet", models)

    def test_multiple_files_different_days(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-29_10.00.00", [_span("gpt-4o", 100, 50, 150)])
            _write_trace(tmp, "2025-11-30_10.00.00", [_span("gpt-4o", 200, 80, 280)])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 2)
            dates = [r["date"] for r in rows]
            self.assertIn("2025-11-29", dates)
            self.assertIn("2025-11-30", dates)

    def test_cache_tokens_extracted(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2025-11-30_10.00.00", [
                _span("claude-3-5-sonnet", 100, 50, 150, cache_read=40, cache_create=10)
            ])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows[0]["cache_read_input_tokens"], 40)
            self.assertEqual(rows[0]["cache_creation_input_tokens"], 10)

    def test_span_without_model_skipped(self):
        with TemporaryDirectory() as tmp:
            span_no_model = {"attributes": {}, "events": [
                {"name": "metadata", "attributes": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
            ]}
            _write_trace(tmp, "2025-11-30_10.00.00", [span_no_model])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_span_without_metadata_event_skipped(self):
        with TemporaryDirectory() as tmp:
            span_no_meta = {
                "attributes": {"entity.2.name": "gpt-4o"},
                "events": [
                    {"name": "data.input", "attributes": {"input": "hello"}},
                ],
            }
            _write_trace(tmp, "2025-11-30_10.00.00", [span_no_meta])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_malformed_json_file_skipped(self):
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "monocle_trace_bad_2025-11-30_10.00.00.json"
            bad.write_text("NOT JSON", encoding="utf-8")
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_file_with_unparseable_timestamp_skipped(self):
        with TemporaryDirectory() as tmp:
            # name doesn't match expected timestamp pattern
            (Path(tmp) / "monocle_trace_nodate.json").write_text(
                json.dumps([_span("gpt-4o", 10, 5, 15)]), encoding="utf-8"
            )
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])


# ---------------------------------------------------------------------------
# Tests: time window filtering
# ---------------------------------------------------------------------------

class TestTimeWindowFiltering(unittest.TestCase):

    def test_today_excludes_old_files(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2020-01-01_10.00.00", [_span("gpt-4o", 100, 50, 150)])
            rows = summarize("today", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_7_days_excludes_old_files(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2020-01-01_10.00.00", [_span("gpt-4o", 100, 50, 150)])
            rows = summarize("7 days", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_15_days_excludes_old_files(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2020-01-01_10.00.00", [_span("gpt-4o", 100, 50, 150)])
            rows = summarize("15 days", monocle_dir=Path(tmp))
            self.assertEqual(rows, [])

    def test_all_includes_old_files(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2020-01-01_10.00.00", [_span("gpt-4o", 100, 50, 150)])
            rows = summarize("all", monocle_dir=Path(tmp))
            self.assertEqual(len(rows), 1)

    def test_unknown_window_treated_as_all(self):
        with TemporaryDirectory() as tmp:
            _write_trace(tmp, "2020-01-01_10.00.00", [_span("gpt-4o", 100, 50, 150)])
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
        rows = [_row()]
        out = format_table(rows)
        for header in ["Date", "Model", "Input", "Cache Read", "Cache Create", "Output", "Total"]:
            self.assertIn(header, out)

    def test_data_present(self):
        rows = [_row()]
        out = format_table(rows)
        self.assertIn("2025-11-30", out)
        self.assertIn("gpt-4o", out)
        self.assertIn("100", out)

    def test_multiple_rows(self):
        rows = [_row("2025-11-29", "gpt-4o"), _row("2025-11-30", "claude-3-5-sonnet")]
        out = format_table(rows)
        self.assertIn("gpt-4o", out)
        self.assertIn("claude-3-5-sonnet", out)
        self.assertIn("2025-11-29", out)
        self.assertIn("2025-11-30", out)

    def test_table_borders_present(self):
        out = format_table([_row()])
        self.assertIn("+", out)
        self.assertIn("|", out)
        self.assertIn("-", out)


def _row(date="2025-11-30", model="gpt-4o"):
    return {
        "date": date,
        "model": model,
        "prompt_tokens": 100,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


if __name__ == "__main__":
    unittest.main()