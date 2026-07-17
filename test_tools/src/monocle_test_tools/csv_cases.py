"""CSV -> fluent test-case adapter for monocle_test_tools.

Loads a flat CSV of okahu-trace test cases and drives them through the
existing fluent TraceAssertion API via @monocle_csv_cases + CsvCase.run.
Config (evaluator, template_path, source=okahu) lives in the test stub;
only per-row data lives in the CSV. See the design doc:
okahu/monocle_backlog_tracking designs/2026-07-17-monocle-csv-testcase-adapter.md
"""
import csv
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from monocle_test_tools.fluent_api import TraceAssertion

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


@dataclass
class CsvCase:
    """One CSV row: an okahu-trace test case driven through the fluent API."""
    case_id: str
    id: str
    workflow_name: str
    fact_name: str = "traces"
    expected: Optional[Union[str, List[str]]] = None
    not_expected: Optional[Union[str, List[str]]] = None
    called_tool: Optional[str] = None
    called_agent: Optional[str] = None
    token_limit: Optional[int] = None
    duration_ms: Optional[float] = None
    has_input: Optional[str] = None
    has_output: Optional[str] = None
    extra_steps: List[dict] = field(default_factory=list)
    notes: Optional[str] = None


def _require(row: dict, column: str, case_id: str, line: int) -> str:
    value = row.get(column, "").strip()
    if value == "":
        raise CsvCaseError(f"line {line} (row '{case_id or '?'}'): required column '{column}' is empty")
    return value


def _validate_extra_steps(steps: List[dict], case_id: str) -> None:
    for step in steps:
        method_name = step["method"]
        method = getattr(TraceAssertion, method_name, None)
        if method is None or not callable(method):
            raise CsvCaseError(
                f"row '{case_id}': extra_json method '{method_name}' is not a TraceAssertion method"
            )
        kwargs = step.get("kwargs", {})
        sig = inspect.signature(method)
        accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_var_kw:
            valid = set(sig.parameters) - {"self"}
            unknown = set(kwargs) - valid
            if unknown:
                raise CsvCaseError(
                    f"row '{case_id}': extra_json '{method_name}' got unknown kwargs {sorted(unknown)}; "
                    f"valid: {sorted(valid)}"
                )


def _row_to_case(row: dict, line: int) -> CsvCase:
    case_id = row.get("case_id", "").strip()
    _require(row, "case_id", case_id, line)
    trace_id = _require(row, "id", case_id, line)
    workflow_name = _require(row, "workflow_name", case_id, line)

    fact_name = row.get("fact_name", "").strip() or "traces"
    if fact_name not in FACT_NAME_VALUES:
        raise CsvCaseError(
            f"row '{case_id}': invalid fact_name '{fact_name}'. Supported: {', '.join(FACT_NAME_VALUES)}"
        )

    extra_steps = parse_extra_json(row.get("extra_json", "").strip(), case_id)
    _validate_extra_steps(extra_steps, case_id)

    case = CsvCase(
        case_id=case_id,
        id=trace_id,
        workflow_name=workflow_name,
        fact_name=fact_name,
        expected=parse_multivalue(row.get("expected", "").strip()),
        not_expected=parse_multivalue(row.get("not_expected", "").strip()),
        called_tool=(row.get("called_tool", "").strip() or None),
        called_agent=(row.get("called_agent", "").strip() or None),
        token_limit=parse_int(row.get("token_limit", "").strip(), "token_limit", case_id),
        duration_ms=parse_float(row.get("duration_ms", "").strip(), "duration_ms", case_id),
        has_input=(row.get("has_input", "").strip() or None),
        has_output=(row.get("has_output", "").strip() or None),
        extra_steps=extra_steps,
        notes=(row.get("notes", "").strip() or None),
    )

    has_assertion = (
        case.expected is not None or case.not_expected is not None
        or case.called_tool is not None or case.called_agent is not None
        or case.token_limit is not None or case.duration_ms is not None
        or case.has_input is not None or case.has_output is not None
        or len(case.extra_steps) > 0
    )
    if not has_assertion:
        raise CsvCaseError(
            f"row '{case_id}': declares a trace source but no assertions (vacuous test). "
            f"Populate at least one of: {', '.join(ASSERTION_COLUMNS)}"
        )
    return case


def load_cases_from_csv(path: str) -> List[CsvCase]:
    """Load and validate a CSV of test cases into a list of CsvCase."""
    if not os.path.isfile(path):
        raise CsvCaseError(f"CSV file not found: {path}")
    rows = read_rows(path)
    if not rows:
        raise CsvCaseError(f"CSV file has no data rows: {path}")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise CsvCaseError(f"{path}: missing required column(s): {', '.join(missing)}")

    cases: List[CsvCase] = []
    seen = set()
    for index, row in enumerate(rows):
        line = index + 2  # +1 for header, +1 for 1-based line numbers
        case = _row_to_case(row, line)
        if case.case_id in seen:
            raise CsvCaseError(f"line {line}: duplicate case_id '{case.case_id}'")
        seen.add(case.case_id)
        cases.append(case)
    return cases
