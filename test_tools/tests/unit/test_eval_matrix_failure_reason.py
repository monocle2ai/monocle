"""A non-pass matrix row must say WHY it failed.

When `evaluate()` raises -- connection reset, read timeout, an empty `result` from
the eval service -- the matrix row previously carried `actual: ""` and an empty
`explanation`, so the reason was only recoverable from the pytest log. Anyone
reading the matrix afterwards (or a stability analyzer consuming it) could not tell
an infrastructure failure from a judge that returned nothing.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from monocle_test_tools.eval_matrix import build_eval_matrix_row
from monocle_test_tools.fluent_api import TraceAssertion

TEMPLATE = {
    "name": "t",
    "eval_prompt": "x",
    "structure_output": {
        "label": {"enums": ["a", "b"], "description": "x"},
        "explanation": {"description": "x"},
    },
}


@pytest.fixture(autouse=True)
def _reset_shared_state():
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None
    TraceAssertion._last_eval = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None
    TraceAssertion._last_eval = None


def test_row_exposes_failure_reason_field():
    last_eval = {
        "trace_id": "abc", "expected": "a", "fact_name": "traces", "label": None,
        "explanation": "", "judge_output": {}, "total_tokens": None,
        "failure_reason": "Failed to reach evaluation service: Connection aborted.",
    }

    row = build_eval_matrix_row(run_id="r", scenario="test_x", last_eval=last_eval, passed=False)

    assert row["status"] == "error"
    assert row["failure_reason"] == "Failed to reach evaluation service: Connection aborted."


def test_row_defaults_failure_reason_empty_when_absent():
    # Passing rows and older stashes carry no reason; the column must still exist so
    # the schema is stable for consumers.
    last_eval = {"trace_id": "abc", "expected": "a", "label": "a", "explanation": "ok",
                 "judge_output": {}, "total_tokens": 5}

    row = build_eval_matrix_row(run_id="r", scenario="test_x", last_eval=last_eval, passed=True)

    assert row["failure_reason"] == ""


def test_check_eval_records_why_evaluate_raised():
    """The reason must be captured inside the library, at the raise site -- not
    scraped from a pytest report -- so it is available to every consumer."""
    span = MagicMock()
    eval_mock = MagicMock()
    eval_mock.evaluate.side_effect = AssertionError(
        "Unexpected response format from evaluation service. Expected 'result' key "
        "in response. Received: {'message': 'Job submitted', 'result': []}")
    asserter = TraceAssertion(filtered_spans=[span], _eval=eval_mock)

    try:
        asserter.check_eval(template=TEMPLATE, expected="a")

        stash = TraceAssertion._last_eval
        assert stash is not None
        assert "Job submitted" in (stash.get("failure_reason") or ""), (
            f"failure reason was not stashed, got {stash.get('failure_reason')!r}")

        row = build_eval_matrix_row(run_id="r", scenario="test_x",
                                    last_eval=stash, passed=False)
        assert row["status"] == "error"
        assert "Job submitted" in row["failure_reason"]
    finally:
        TraceAssertion._assertion_errors = []


def test_judge_mismatch_row_still_carries_the_judge_explanation():
    """A judge disagreement is not an infrastructure failure: evaluate() returned a
    label, so the row stays self-describing via actual + explanation and needs no
    failure_reason."""
    span = MagicMock()
    eval_mock = MagicMock()
    eval_mock.evaluate.return_value = ("b", "the judge reasoned thus")
    asserter = TraceAssertion(filtered_spans=[span], _eval=eval_mock)

    try:
        asserter.check_eval(template=TEMPLATE, expected="a")

        row = build_eval_matrix_row(run_id="r", scenario="test_x",
                                    last_eval=TraceAssertion._last_eval, passed=False)
        assert row["status"] == "fail"
        assert row["actual"] == "b"
        assert row["explanation"] == "the judge reasoned thus"
        assert row["failure_reason"] == ""
    finally:
        TraceAssertion._assertion_errors = []


def test_failure_reason_survives_json_serialisation():
    # The recorder writes with json.dump(..., default=str); the reason must round-trip.
    last_eval = {"trace_id": "abc", "expected": "a", "label": None, "explanation": "",
                 "judge_output": {}, "total_tokens": None,
                 "failure_reason": "Fact map request timed out: read timeout=60"}
    row = build_eval_matrix_row(run_id="r", scenario="test_x", last_eval=last_eval, passed=False)

    assert json.loads(json.dumps({"records": [row]}, default=str))["records"][0][
        "failure_reason"] == "Fact map request timed out: read timeout=60"
