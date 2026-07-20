"""Unit tests for the filter-mode `check_eval` argument validation.

These exercise the validation branches that raise *before* any network/env access,
plus the end-to-end wiring of `with_trace_source("okahu", start_time=, end_time=)` +
`check_eval(...)` (filter mode) with OkahuFilteredEval mocked out.
No live HTTP and no ML deps are touched.
"""
from unittest.mock import MagicMock, patch

import pytest

from monocle_test_tools.fluent_api import TraceAssertion


@pytest.fixture(autouse=True)
def _reset_trace_assertion_class_state():
    """Reset shared class-level TraceAssertion state before AND after each test.

    Several tests here build a `TraceAssertion()` directly (bypassing the
    `monocle_trace_asserter` fixture, whose teardown calls `cleanup()`), and some
    intentionally leave a recorded assertion or a stashed `_eval_report` behind.
    A manual reset as the LAST statement of a test body is fragile: if an earlier
    assertion in that same test body fails, the reset is skipped and the dirty
    class-level state (`_assertion_errors` in particular) bleeds into whichever
    test runs next -- pytest_plugin.py's `pytest_runtest_makereport` flips any
    passing test to failed when `TraceAssertion().has_assertions()` is true.
    An autouse fixture with both a setup and a teardown reset closes that gap.
    """
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None


def _asserter():
    return TraceAssertion()


def _make_span_asserter(eval_result=("no_hallucination", "ok")):
    """Span-mode asserter: one span, mocked evaluator, NO `_okahu_filter` set.

    Mirrors `_make_asserter` in test_check_eval_template_path.py's
    `test_template_dict_selector_reaches_evaluate_unmodified`.
    """
    span = MagicMock()
    eval_mock = MagicMock()
    eval_mock.evaluate.return_value = eval_result
    return TraceAssertion(filtered_spans=[span], _eval=eval_mock)


def test_check_eval_filter_mode_requires_exactly_one_selector():
    a = _asserter().with_trace_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    with pytest.raises(ValueError):
        a.check_eval(expected="no_hallucination")                          # none
    with pytest.raises(ValueError):
        a.check_eval(eval_name="hallucination", template={"name": "x"},
                     expected="no_hallucination")                          # two


def test_check_eval_filter_mode_requires_expectation():
    a = _asserter().with_trace_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    with pytest.raises(ValueError):
        a.check_eval(eval_name="hallucination")


def test_check_eval_filter_mode_rejects_dict_expected():
    a = _asserter().with_trace_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    with pytest.raises(ValueError):
        a.check_eval(eval_name="hallucination", expected={"aa": "no_hallucination"})


def test_check_eval_filter_mode_rejects_overlapping_expected_not_expected():
    a = _asserter().with_trace_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    with pytest.raises(ValueError):
        a.check_eval(eval_name="hallucination",
                     expected="x", not_expected=["x", "y"])


def _report(passed=1, failed=0, errors=0, scenarios=None):
    return {"job_id": "job-1", "status": "completed",
            "summary": {"total": passed + failed + errors, "passed": passed,
                        "failed": failed, "errors": errors, "duration_seconds": 1},
            "scenarios": scenarios or [
                {"fact_id": "aa", "expected": ["no_hallucination"], "actual": "no_hallucination",
                 "status": "pass", "job_id": "job-1", "explanation": "", "workflow": "wf"}]}


def test_with_trace_source_okahu_window_records_scope_no_import():
    a = _asserter()
    with patch.object(a.validator, "import_traces") as imp:
        a.with_trace_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    imp.assert_not_called()
    assert a._okahu_filter == {"workflows": ["wf"], "start_time": "s",
                               "end_time": "e", "fact_name": "traces"}


def test_with_trace_source_okahu_window_accepts_workflow_list():
    a = _asserter().with_trace_source("okahu", workflow_name=["a", "b"],
                                      start_time="s", end_time="e")
    assert a._okahu_filter["workflows"] == ["a", "b"]


def test_with_trace_source_okahu_id_and_window_conflict():
    with pytest.raises(ValueError):
        _asserter().with_trace_source("okahu", id="t1", workflow_name="wf",
                                      start_time="s", end_time="e")


def test_with_trace_source_okahu_window_requires_workflow_and_bounds():
    with pytest.raises(ValueError):
        _asserter().with_trace_source("okahu", start_time="s", end_time="e")  # no workflow
    with pytest.raises(ValueError):
        _asserter().with_trace_source("okahu", workflow_name="wf", start_time="s")  # no end


def test_with_trace_source_window_rejected_for_local_and_file():
    with pytest.raises(ValueError):
        _asserter().with_trace_source("local", start_time="s", end_time="e")
    with pytest.raises(ValueError):
        _asserter().with_trace_source("file", start_time="s", end_time="e")


def test_with_trace_source_okahu_id_still_imports():
    a = _asserter()
    with patch.object(a.validator, "import_traces") as imp:
        a.with_trace_source("okahu", id="t1", workflow_name="wf")
    imp.assert_called_once()
    assert a._okahu_filter is None


def test_check_eval_filter_mode_runs_filtered_flow_and_stashes_report(monkeypatch):
    monkeypatch.delenv("MONOCLE_EVAL_MATRIX", raising=False)
    a = _asserter().with_trace_source("okahu", workflow_name="wf",
                                      start_time="s", end_time="e")
    with patch("monocle_test_tools.evals.okahu_filtered_eval.OkahuFilteredEval.from_env") as mk:
        mk.return_value.run_filtered.return_value = _report()
        result = a.check_eval(eval_name="hallucination", expected="no_hallucination", min_facts=1)
    assert result.get_eval_report()["job_id"] == "job-1"
    mk.return_value.run_filtered.assert_called_once()


def test_check_eval_filter_mode_records_failure_on_fail_over_threshold(monkeypatch):
    monkeypatch.delenv("MONOCLE_EVAL_MATRIX", raising=False)
    fail_scn = [{"fact_id": "bb", "expected": ["no_hallucination"], "actual": "major_hallucination",
                 "status": "fail", "job_id": "job-1", "explanation": "", "workflow": "wf"}]
    a = _asserter().with_trace_source("okahu", workflow_name="wf", start_time="s", end_time="e")
    # This test intentionally leaves a recorded assertion behind to inspect it
    # below. pytest_plugin.py's pytest_runtest_makereport hook checks
    # `TraceAssertion().has_assertions()` right after the test *call* phase
    # finishes -- before the autouse fixture's teardown-side reset (which only
    # runs in the later teardown phase) has a chance to run -- so the cleanup
    # must happen inside the test body. The try/finally (rather than a bare
    # last-statement reset) guarantees it runs even if an assertion above it
    # fails, so a real regression here can't also contaminate whichever test
    # runs next.
    try:
        with patch("monocle_test_tools.evals.okahu_filtered_eval.OkahuFilteredEval.from_env") as mk:
            mk.return_value.run_filtered.return_value = _report(passed=0, failed=1, scenarios=fail_scn)
            result = a.check_eval(eval_name="hallucination", expected="no_hallucination")
        # @collect_assertions records the gate failure rather than raising inline.
        assert result.has_assertions()
        assert result.get_eval_report()["summary"]["failed"] == 1
    finally:
        TraceAssertion._assertion_errors = []


def test_check_eval_filter_only_params_rejected_in_span_mode():
    a = _asserter()
    a._filtered_spans = ["not-empty"]  # span mode (no _okahu_filter)
    with pytest.raises(ValueError):
        a.check_eval(eval_name="hallucination", expected="x", min_facts=5)
    with pytest.raises(ValueError):
        a.check_eval(eval_name="hallucination", expected="x", fail_threshold=2)


def test_get_eval_report_uniform_filter_mode(monkeypatch, tmp_path):
    # Filter-mode: run_filtered's report is stashed as-is via check_eval (mocked client).
    a = _asserter()
    a._filtered_spans = None
    a._okahu_filter = {"workflows": ["wf"], "start_time": "s", "end_time": "e", "fact_name": "traces"}
    with patch("monocle_test_tools.evals.okahu_filtered_eval.OkahuFilteredEval.from_env") as mk:
        mk.return_value.run_filtered.return_value = _report()
        a.check_eval(eval_name="hallucination", expected="no_hallucination")
    report = a.get_eval_report()
    assert report["summary"]["passed"] == 1
    assert a.get_eval_failures() == []
    out = tmp_path / "r.json"
    a.write_eval_report(str(out))
    assert out.is_file()


def test_check_eval_span_mode_pass_stashes_uniform_report():
    """Real span-mode (no `_okahu_filter`): matching `expected` -> report is a 1-fact
    uniform report (same shape as filter mode) with no failures."""
    a = _make_span_asserter(eval_result=("no_hallucination", "looks fine"))

    result = a.check_eval(eval_name="hallucination", expected="no_hallucination")

    assert not result.has_assertions(), result.get_assertion_messages()
    report = result.get_eval_report()
    assert report is not None
    assert report["summary"]["total"] == 1
    assert result.get_eval_failures() == []


def test_check_eval_span_mode_fail_still_stashes_uniform_report():
    """Important 1 regression test: a FAILING span-mode eval must still populate
    the uniform report (pass or fail, the report is always stashed), not just
    record the assertion. Before the fix, this failed because the span-mode
    `_eval_report` assignment ran only after (and thus never reached on) the
    pass/fail raise.

    Uses try/finally (see the comment in
    test_check_eval_filter_mode_records_failure_on_fail_over_threshold above)
    since this test intentionally leaves a recorded assertion behind to inspect.
    """
    a = _make_span_asserter(eval_result=("major_hallucination", "not fine"))

    try:
        result = a.check_eval(eval_name="hallucination", expected="no_hallucination")

        # @collect_assertions suppresses the AssertionError and records it instead.
        assert result.has_assertions()
        report = result.get_eval_report()
        assert report is not None
        assert report["summary"]["total"] == 1
        failures = result.get_eval_failures()
        assert len(failures) == 1
    finally:
        TraceAssertion._assertion_errors = []
