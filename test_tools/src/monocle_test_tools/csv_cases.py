"""CSV -> fluent eval test-case adapter for monocle_test_tools.

Loads a flat CSV of okahu-trace *evaluation* test cases and drives each row
through the existing fluent TraceAssertion API via ``@monocle_csv_cases`` +
``CsvCase.run``, so eval test data can be curated in a spreadsheet.

Scope (v0): evaluation tests only. Each row asserts an eval label
(``expected``/``not_expected`` via ``check_eval``) and, optionally, operational
guard rails on the run: a token budget (``max_tokens`` -> ``under_token_limit``)
and an effort ceiling (``max_duration_ms`` -> ``under_duration``). The loader
bridges exactly these three stable, eval-relevant fluent methods -- not the
open set of assertions. General non-eval assertions (tool/agent calls,
input/output, arbitrary fluent methods) and multi-condition rows are
intentionally out of scope for v0; see the PR for the rationale and future
considerations.

Config-in-code, data-in-CSV: the test stub owns everything constant across the
sheet (the evaluator via ``with_evaluation(...)``, the eval ``template_path``,
and the trace source, fixed to ``okahu`` in this version); the CSV owns only
what varies per row (fact id, workflow, expected labels, guard rails).
One row = one test.

Example test stub::

    @monocle_csv_cases("cases.csv")
    def test_cases(monocle_trace_asserter, case):
        case.run(monocle_trace_asserter.with_evaluation("okahu"),
                 template_path=TEMPLATE_PATH)
"""
import csv
import inspect
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pytest

REQUIRED_COLUMNS = ("case_id", "fact_id", "workflow_name")


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


@dataclass
class CsvCase:
    """One CSV row: an okahu-trace evaluation test case driven through the fluent API."""
    case_id: str
    fact_id: str
    workflow_name: str
    fact_name: str = "traces"
    expected: Optional[Union[str, List[str]]] = None
    not_expected: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_duration_ms: Optional[float] = None
    notes: Optional[str] = None

    def run(self, asserter, **check_eval_kwargs) -> None:
        """Drive this row through the fluent TraceAssertion `asserter`.

        Runs the eval (``check_eval``) plus optional operational guard rails
        (token budget, duration). Independent assertions are each invoked on
        the base asserter (not threaded): PR #721 collects failures on the
        asserter and the pytest plugin reports them. Only misconfiguration
        (an eval row with no evaluator) raises from here.
        """
        asserter.with_trace_source("okahu", id=self.fact_id, workflow_name=self.workflow_name)

        if getattr(asserter, "_eval", None) is None:
            raise CsvCaseError(
                f"row '{self.case_id}': is an evaluation case but no evaluator is "
                f"configured. Add .with_evaluation(...) to your test stub."
            )
        asserter.check_eval(
            expected=self.expected,
            not_expected=self.not_expected,
            fact_name=self.fact_name,
            **check_eval_kwargs,
        )

        if self.max_tokens is not None:
            asserter.under_token_limit(self.max_tokens)
        if self.max_duration_ms is not None:
            asserter.under_duration(self.max_duration_ms, units="ms")


def _require(row: dict, column: str, case_id: str, line: int) -> str:
    value = row.get(column, "").strip()
    if value == "":
        raise CsvCaseError(f"line {line} (row '{case_id or '?'}'): required column '{column}' is empty")
    return value


def _row_to_case(row: dict, line: int) -> CsvCase:
    case_id = row.get("case_id", "").strip()
    _require(row, "case_id", case_id, line)
    fact_id = _require(row, "fact_id", case_id, line)
    workflow_name = _require(row, "workflow_name", case_id, line)

    fact_name = row.get("fact_name", "").strip() or "traces"

    expected = parse_multivalue(row.get("expected", "").strip())
    not_expected = parse_multivalue(row.get("not_expected", "").strip())
    if expected is None and not_expected is None:
        raise CsvCaseError(
            f"row '{case_id}': an evaluation case requires an 'expected' or "
            f"'not_expected' label (a token/duration guard rail alone is not a test)."
        )

    return CsvCase(
        case_id=case_id,
        fact_id=fact_id,
        workflow_name=workflow_name,
        fact_name=fact_name,
        expected=expected,
        not_expected=not_expected,
        max_tokens=parse_int(row.get("max_tokens", "").strip(), "max_tokens", case_id),
        max_duration_ms=parse_float(row.get("max_duration_ms", "").strip(), "max_duration_ms", case_id),
        notes=(row.get("notes", "").strip() or None),
    )


def load_cases_from_csv(path: str) -> List[CsvCase]:
    """Load and validate a CSV of eval test cases into a list of CsvCase."""
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


def monocle_csv_cases(path: str):
    """Parametrize a test over the rows of a CSV of eval CsvCase.

    Usage:
        @monocle_csv_cases("cases.csv")
        def test_cases(monocle_trace_asserter, case):
            case.run(monocle_trace_asserter.with_evaluation("okahu"),
                     template_path=TEMPLATE_PATH)

    Relative paths resolve against the calling test file's directory, so the
    test works regardless of pytest's invocation directory.
    """
    resolved = path
    if not os.path.isabs(path):
        caller_file = inspect.stack()[1].filename
        resolved = os.path.join(os.path.dirname(os.path.abspath(caller_file)), path)
    cases = load_cases_from_csv(resolved)
    return pytest.mark.parametrize("case", cases, ids=[c.case_id for c in cases])
