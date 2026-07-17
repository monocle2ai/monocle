from unittest.mock import Mock, patch
from monocle_test_tools import eval_matrix
from monocle_test_tools.eval_matrix import build_eval_matrix_row, reset_records, record_eval_row_for, get_records
from monocle_test_tools.fluent_api import TraceAssertion


def test_build_eval_matrix_row_pass_includes_tokens():
    last_eval = {
        "trace_id": "abc123",
        "expected": ["no_hallucination"],
        "fact_name": "traces",
        "label": "no_hallucination",
        "explanation": "matches expectations",
        "judge_output": {
            "claim_verdicts": [{"claim": "x", "verdict": "supported"}],
            "hallucination_types": [],
            "entity_match_check": "ok",
        },
        "total_tokens": 123,
    }

    row = build_eval_matrix_row(run_id="run-1", scenario="test_x", last_eval=last_eval, passed=True)

    assert row == {
        "run_id": "run-1",
        "scenario": "test_x",
        "trace_id": "abc123",
        "expected": ["no_hallucination"],
        "actual": "no_hallucination",
        "status": "pass",
        "explanation": "matches expectations",
        "total_tokens": 123,
        "claim_verdicts": [{"claim": "x", "verdict": "supported"}],
        "hallucination_types": [],
        "entity_match_check": "ok",
        # Filtered-flow columns default to empty for interactive rows (additive widening).
        "fact_id": "",
        "workflow": "",
        "job_id": "",
    }


def test_build_eval_matrix_row_not_passed_with_label_is_drift():
    last_eval = {
        "trace_id": "def456",
        "expected": ["no_hallucination"],
        "fact_name": "traces",
        "label": "major_hallucination",
        "explanation": "diverged",
        "judge_output": {},
        "total_tokens": 42,
    }

    row = build_eval_matrix_row(run_id="run-2", scenario="test_y", last_eval=last_eval, passed=False)

    assert row["status"] == "drift"
    assert row["actual"] == "major_hallucination"
    assert row["claim_verdicts"] == []
    assert row["hallucination_types"] == []
    assert row["entity_match_check"] == ""


def test_build_eval_matrix_row_not_passed_without_label_is_error():
    last_eval = {
        "trace_id": "",
        "expected": None,
        "fact_name": "traces",
        "label": None,
        "explanation": "",
        "judge_output": {},
        "total_tokens": None,
    }

    row = build_eval_matrix_row(run_id="run-3", scenario="test_z", last_eval=last_eval, passed=False)

    assert row["status"] == "error"
    assert row["actual"] == ""
    assert row["total_tokens"] is None
    assert row["trace_id"] == ""


def test_build_eval_matrix_row_is_none_safe_for_missing_judge_output_keys():
    last_eval = {
        "trace_id": "abc",
        "expected": "no_hallucination",
        "fact_name": "traces",
        "label": "no_hallucination",
        "explanation": None,
        "judge_output": None,
        "total_tokens": None,
    }

    row = build_eval_matrix_row(run_id="run-4", scenario="test_w", last_eval=last_eval, passed=True)

    assert row["status"] == "pass"
    assert row["claim_verdicts"] == []
    assert row["hallucination_types"] == []
    assert row["entity_match_check"] == ""
    # explanation is passed through as-is from last_eval (no forced fallback)
    assert row["explanation"] is None


def test_recorder_does_not_bleed_stale_eval_between_tests():
    """Regression test: verify that _last_eval state doesn't leak from scenario A to scenario B.

    This tests the fragile class-level _last_eval attribute isolation: if a test
    never populated _last_eval, it should record NO row and NOT inherit a previous
    scenario's _last_eval.
    """
    reset_records()

    # Scenario A: test with _last_eval set
    asserter_a = TraceAssertion.get_trace_asserter()
    # At this point, cleanup() has been called, so _last_eval is None
    assert TraceAssertion._last_eval is None

    # Manually set _last_eval (simulating what check_eval would do)
    TraceAssertion._last_eval = {
        "trace_id": "t1",
        "expected": "major_hallucination",
        "fact_name": "traces",
        "label": "major_hallucination",
        "explanation": "e",
        "judge_output": {},
        "total_tokens": 10,
    }

    # Record scenario A
    mock_config_a = Mock()
    mock_config_a.getoption.return_value = "test-matrix.json"
    mock_node_a = Mock()
    mock_node_a.name = "test_scenario_a"
    mock_node_a.callspec = None
    mock_rep_call_a = Mock()
    mock_rep_call_a.passed = True
    mock_node_a.rep_call = mock_rep_call_a
    mock_request_a = Mock()
    mock_request_a.config = mock_config_a
    mock_request_a.node = mock_node_a

    record_eval_row_for(mock_config_a, mock_request_a, asserter_a)

    # Cleanup A (resets _last_eval to None)
    asserter_a.cleanup()
    assert TraceAssertion._last_eval is None

    # Scenario B: test without _last_eval set
    asserter_b = TraceAssertion.get_trace_asserter()
    # Cleanup has reset _last_eval, so it should be None
    assert TraceAssertion._last_eval is None

    # Do NOT set _last_eval for scenario B

    # Try to record scenario B
    mock_config_b = Mock()
    mock_config_b.getoption.return_value = "test-matrix.json"
    mock_node_b = Mock()
    mock_node_b.name = "test_scenario_b"
    mock_node_b.callspec = None
    mock_rep_call_b = Mock()
    mock_rep_call_b.passed = True
    mock_node_b.rep_call = mock_rep_call_b
    mock_request_b = Mock()
    mock_request_b.config = mock_config_b
    mock_request_b.node = mock_node_b

    record_eval_row_for(mock_config_b, mock_request_b, asserter_b)

    # Cleanup B
    asserter_b.cleanup()

    # Verify: exactly ONE row recorded (from scenario A only)
    records = get_records()
    assert len(records) == 1, f"Expected 1 record, got {len(records)}: {records}"

    # Verify the one row is from scenario A
    row_a = records[0]
    assert row_a["scenario"] == "test_scenario_a"
    assert row_a["trace_id"] == "t1"
    assert row_a["actual"] == "major_hallucination"
    assert row_a["total_tokens"] == 10


# --- Task 7: widened schema + filtered-report recorder bridge ----------------

def test_build_eval_matrix_row_carries_optional_filtered_fields():
    last_eval = {"trace_id": "aa", "fact_id": "aa", "workflow": "wf", "job_id": "job-1",
                 "expected": ["no_hallucination"], "label": "no_hallucination",
                 "explanation": "ok", "judge_output": {}, "total_tokens": 5}
    row = build_eval_matrix_row(run_id="r", scenario="aa", last_eval=last_eval, passed=True)
    assert row["fact_id"] == "aa" and row["workflow"] == "wf" and row["job_id"] == "job-1"
    assert row["status"] == "pass"


def test_build_eval_matrix_row_defaults_filtered_fields_empty_for_interactive():
    last_eval = {"trace_id": "t", "expected": "x", "label": "x", "explanation": "",
                 "judge_output": {}, "total_tokens": None}
    row = build_eval_matrix_row(run_id="r", scenario="test_x", last_eval=last_eval, passed=True)
    assert row["fact_id"] == "" and row["workflow"] == "" and row["job_id"] == ""


def test_record_eval_rows_from_report_appends_one_row_per_scenario_when_enabled(monkeypatch):
    reset_records()
    monkeypatch.setenv("MONOCLE_EVAL_MATRIX", "1")
    report = {"job_id": "job-1", "scenarios": [
        {"fact_id": "aa", "workflow": "wf", "job_id": "job-1", "expected": ["no_hallucination"],
         "actual": "no_hallucination", "status": "pass", "explanation": "ok"},
        {"fact_id": "bb", "workflow": "wf", "job_id": "job-1", "expected": ["no_hallucination"],
         "actual": "major_hallucination", "status": "drift", "explanation": "x"}]}
    eval_matrix.record_eval_rows_from_report(report)
    rows = get_records()
    assert [r["fact_id"] for r in rows] == ["aa", "bb"]
    assert [r["status"] for r in rows] == ["pass", "drift"]


def test_record_eval_rows_from_report_self_skips_when_disabled(monkeypatch):
    reset_records()
    monkeypatch.delenv("MONOCLE_EVAL_MATRIX", raising=False)
    eval_matrix.record_eval_rows_from_report({"scenarios": [{"fact_id": "aa", "status": "pass"}]})
    assert get_records() == []
