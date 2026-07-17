"""CSV -> fluent test-case adapter for monocle_test_tools.

Loads a flat CSV of okahu-trace test cases and drives them through the
existing fluent TraceAssertion API via @monocle_csv_cases + CsvCase.run.
Config (evaluator, template_path, source=okahu) lives in the test stub;
only per-row data lives in the CSV. See the design doc:
okahu/monocle_backlog_tracking designs/2026-07-17-monocle-csv-testcase-adapter.md
"""
import csv
import json
from typing import Dict, List, Optional, Union

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


def parse_multivalue(cell: str) -> Optional[Union[str, List[str]]]:
    """Parse a cell into None, a string, or a list (JSON array or pipe-delimited)."""
    if cell == "":
        return None
    if cell.startswith("["):
        try:
            value = json.loads(cell)
        except json.JSONDecodeError:
            return cell  # not JSON; treat as a literal string
        if isinstance(value, list):
            return [str(v) for v in value]
        return cell
    if "|" in cell:
        return [part.strip() for part in cell.split("|") if part.strip() != ""]
    return cell


def parse_int(cell: str, column: str, case_id: str) -> Optional[int]:
    if cell == "":
        return None
    try:
        return int(cell)
    except ValueError:
        raise CsvCaseError(f"row '{case_id}': column '{column}' must be an integer, got '{cell}'")


def parse_float(cell: str, column: str, case_id: str) -> Optional[float]:
    if cell == "":
        return None
    try:
        return float(cell)
    except ValueError:
        raise CsvCaseError(f"row '{case_id}': column '{column}' must be a number, got '{cell}'")


def parse_extra_json(cell: str, case_id: str) -> List[dict]:
    """Parse the extra_json cell into a list of {"method", "kwargs"?} steps."""
    if cell == "":
        return []
    try:
        value = json.loads(cell)
    except json.JSONDecodeError as exc:
        raise CsvCaseError(f"row '{case_id}': column 'extra_json' is not valid JSON — {exc}")
    if not isinstance(value, list):
        raise CsvCaseError(f"row '{case_id}': column 'extra_json' must be a JSON array of steps")
    for step in value:
        if not isinstance(step, dict) or "method" not in step or not isinstance(step["method"], str):
            raise CsvCaseError(
                f"row '{case_id}': each extra_json step must be an object with a string 'method'"
            )
        if "kwargs" in step and not isinstance(step["kwargs"], dict):
            raise CsvCaseError(f"row '{case_id}': extra_json step 'kwargs' must be an object")
    return value
