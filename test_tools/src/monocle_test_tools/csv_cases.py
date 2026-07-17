"""CSV -> fluent test-case adapter for monocle_test_tools.

Loads a flat CSV of okahu-trace test cases and drives them through the
existing fluent TraceAssertion API via @monocle_csv_cases + CsvCase.run.
Config (evaluator, template_path, source=okahu) lives in the test stub;
only per-row data lives in the CSV. See the design doc:
okahu/monocle_backlog_tracking designs/2026-07-17-monocle-csv-testcase-adapter.md
"""
import csv
from typing import Dict, List

REQUIRED_COLUMNS = ("case_id", "id", "workflow_name")

FACT_NAME_VALUES = (
    "traces", "inferences", "agentic_turns", "agentic_sessions",
    "agent_invocation", "tool_execution", "commits", "conversations",
    "test_runs", "tests",
)

# Columns that constitute an assertion; a valid row has >= 1 populated.
ASSERTION_COLUMNS = (
    "expected", "not_expected", "called_tool", "called_agent",
    "token_limit", "duration_ms", "has_input", "has_output", "extra_json",
)


class CsvCaseError(ValueError):
    """Raised for invalid CSV content; message carries path/line context."""


def read_rows(path: str) -> List[Dict[str, str]]:
    """Read a CSV file into a list of {column: value} dicts (values stripped)."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: List[Dict[str, str]] = []
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if key is None:
                    continue
                row[key.strip()] = (value or "").strip()
            rows.append(row)
        return rows
