"""Unit tests for the filtered-eval fluent API argument validation.

These exercise the validation branches that raise *before* any network/env access,
plus the end-to-end wiring of check_eval_filtered with OkahuFilteredEval mocked out.
No live HTTP and no ML deps are touched.
"""
from unittest.mock import patch

import pytest

from monocle_test_tools.fluent_api import TraceAssertion


def _asserter():
    return TraceAssertion()


def test_with_filtered_source_records_scope():
    a = _asserter().with_filtered_source("okahu", workflow_name="wf",
                                         start_time="s", end_time="e")
    assert a._filtered_scope == {"workflows": ["wf"], "start_time": "s",
                                 "end_time": "e", "fact_name": "traces"}


def test_with_filtered_source_accepts_workflow_list():
    a = _asserter().with_filtered_source("okahu", workflow_name=["a", "b"],
                                         start_time="s", end_time="e")
    assert a._filtered_scope["workflows"] == ["a", "b"]


def test_with_filtered_source_rejects_non_okahu():
    with pytest.raises(ValueError):
        _asserter().with_filtered_source("local", workflow_name="wf",
                                         start_time="s", end_time="e")


def test_check_eval_filtered_requires_source():
    with pytest.raises(AssertionError):
        _asserter().check_eval_filtered(eval_name="hallucination", expected="no_hallucination")


def test_check_eval_filtered_requires_exactly_one_selector():
    a = _asserter().with_filtered_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    with pytest.raises(ValueError):
        a.check_eval_filtered(expected="no_hallucination")                          # none
    with pytest.raises(ValueError):
        a.check_eval_filtered(eval_name="hallucination", template={"name": "x"},
                              expected="no_hallucination")                          # two


def test_check_eval_filtered_requires_expectation():
    a = _asserter().with_filtered_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    with pytest.raises(ValueError):
        a.check_eval_filtered(eval_name="hallucination")


def test_check_eval_filtered_rejects_dict_expected():
    a = _asserter().with_filtered_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    with pytest.raises(ValueError):
        a.check_eval_filtered(eval_name="hallucination", expected={"aa": "no_hallucination"})


def test_check_eval_filtered_rejects_overlapping_expected_not_expected():
    a = _asserter().with_filtered_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    with pytest.raises(ValueError):
        a.check_eval_filtered(eval_name="hallucination",
                              expected="x", not_expected=["x", "y"])


def _report(passed=1, drift=0, errors=0, scenarios=None):
    return {"job_id": "job-1", "status": "completed",
            "summary": {"total": passed + drift + errors, "passed": passed,
                        "drift": drift, "errors": errors, "duration_seconds": 1},
            "scenarios": scenarios or [
                {"fact_id": "aa", "expected": ["no_hallucination"], "actual": "no_hallucination",
                 "status": "pass", "job_id": "job-1", "explanation": "", "workflow": "wf"}]}


def test_check_eval_filtered_passes_and_stashes_report(monkeypatch):
    monkeypatch.delenv("MONOCLE_EVAL_MATRIX", raising=False)  # keep the recorder quiet
    a = _asserter().with_filtered_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    with patch("monocle_test_tools.evals.okahu_filtered_eval.OkahuFilteredEval.from_env") as mk:
        mk.return_value.run_filtered.return_value = _report()
        result = a.check_eval_filtered(eval_name="hallucination", expected="no_hallucination")
    assert result is a
    assert a.get_filtered_eval_report()["job_id"] == "job-1"
    assert a.get_filtered_eval_failures() == []


def test_check_eval_filtered_raises_on_drift_over_threshold(monkeypatch):
    monkeypatch.delenv("MONOCLE_EVAL_MATRIX", raising=False)
    drift_scn = [{"fact_id": "bb", "expected": ["no_hallucination"], "actual": "major_hallucination",
                  "status": "drift", "job_id": "job-1", "explanation": "", "workflow": "wf"}]
    a = _asserter().with_filtered_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    with patch("monocle_test_tools.evals.okahu_filtered_eval.OkahuFilteredEval.from_env") as mk:
        mk.return_value.run_filtered.return_value = _report(passed=0, drift=1, scenarios=drift_scn)
        with pytest.raises(AssertionError):
            a.check_eval_filtered(eval_name="hallucination", expected="no_hallucination")
    # report + failures are still available after the raise
    assert a.get_filtered_eval_report()["summary"]["drift"] == 1
    assert [f["fact_id"] for f in a.get_filtered_eval_failures()] == ["bb"]
