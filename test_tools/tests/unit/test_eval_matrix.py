from monocle_test_tools.eval_matrix import build_eval_matrix_row


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
