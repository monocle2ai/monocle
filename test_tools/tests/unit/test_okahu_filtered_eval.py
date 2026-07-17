"""Unit tests for the filtered-eval helpers and orchestration.

All network calls are mocked with unittest.mock (the repo convention; see
test_okahu_eval_http_errors.py) — no live HTTP and no ML deps are exercised here.
"""
from monocle_test_tools.evals import okahu_filtered_eval as feval


# --- Task 1: fact_id normalization + label detection -------------------------

def test_normalize_strips_0x_prefix():
    assert feval.normalize_fact_id("0x642dab") == "642dab"
    assert feval.normalize_fact_id("642dab") == "642dab"
    assert feval.normalize_fact_id(123) == "123"


def test_has_label_true_only_when_eval_found_and_label_present():
    assert feval.has_label({"eval_found": True, "eval_result": {"label": "x"}}) is True
    assert feval.has_label({"eval_found": True, "eval_result": {}}) is False
    assert feval.has_label({"eval_found": False, "eval_result": {"label": "x"}}) is False
    assert feval.has_label(None) is False
