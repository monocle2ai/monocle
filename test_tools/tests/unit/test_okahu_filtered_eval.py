"""Unit tests for the filtered-eval helpers and orchestration.

All network calls are mocked with unittest.mock (the repo convention; see
test_okahu_eval_http_errors.py) — no live HTTP and no ML deps are exercised here.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from monocle_test_tools.evals import okahu_filtered_eval as feval
from monocle_test_tools.evals.okahu_filtered_eval import OkahuFilteredEval


def _client():
    return OkahuFilteredEval(api_key="k", eval_base="https://eval.example/api",
                             api_base="https://api.example")


def _resp(payload, status=200):
    """A stand-in requests.Response with .json() and a no-op raise_for_status()."""
    resp = MagicMock()
    resp.json.return_value = payload
    resp.status_code = status
    resp.raise_for_status.return_value = None
    return resp


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


# --- Task 2: blanket grading + filtered report assembly (job_id-gated) --------

def _row(fid, label, job_id="job-1", finish_type=None, workflow="wf"):
    return {"fact_id": fid, "job_id": job_id, "eval_found": True, "workflow": workflow,
            "eval_result": {"label": label, "explanation": "why"},
            "finish_type": finish_type}


def test_grade_label_any_of_and_not_expected():
    assert feval.grade_label("minor_hallucination", accepted=["minor_hallucination", "no_hallucination"]) == "pass"
    assert feval.grade_label("major_hallucination", accepted=["no_hallucination"]) == "fail"
    assert feval.grade_label("major_hallucination", not_expected=["minor_hallucination", "major_hallucination"]) == "fail"
    assert feval.grade_label("no_hallucination", not_expected=["major_hallucination"]) == "pass"
    assert feval.grade_label(None, accepted=["no_hallucination"]) == "missing"


def test_filtered_report_grades_every_returned_fact_and_gates_on_job_id():
    results = [_row("aa", "no_hallucination", job_id="job-1"),
               _row("bb", "major_hallucination", job_id="job-1"),
               _row("cc", "no_hallucination", job_id="OLD")]        # stale -> gated out entirely
    report = feval.build_filtered_report("no_hallucination", None, results, job_id="job-1")
    by_fid = {s["fact_id"]: s["status"] for s in report["scenarios"]}
    assert by_fid == {"aa": "pass", "bb": "fail"}                   # cc excluded (wrong job)
    assert report["summary"] == {"total": 2, "passed": 1, "failed": 1,
                                 "errors": 0, "duration_seconds": None}


def test_filtered_report_marks_error_on_finish_type_error():
    results = [{"fact_id": "aa", "job_id": "job-1", "eval_found": True,
                "eval_result": {}, "finish_type": "error", "workflow": "wf"}]
    report = feval.build_filtered_report("no_hallucination", None, results, job_id="job-1")
    assert report["scenarios"][0]["status"] == "error"
    assert report["summary"]["errors"] == 1


def test_filtered_report_normalizes_0x_ids():
    results = [_row("0xaa", "no_hallucination", job_id="job-1")]
    report = feval.build_filtered_report("no_hallucination", None, results, job_id="job-1")
    assert report["scenarios"][0]["fact_id"] == "aa"
    assert report["scenarios"][0]["status"] == "pass"


def test_filtered_report_not_expected_only():
    results = [_row("aa", "major_hallucination", job_id="job-1")]
    report = feval.build_filtered_report(None, ["minor_hallucination", "major_hallucination"], results, job_id="job-1")
    assert report["scenarios"][0]["status"] == "fail"


# --- Task 3: labeled-coverage set (poller support) ---------------------------

def test_labeled_fact_ids_gated_by_job_id():
    results = [_row("aa", "x", job_id="job-1"),
               _row("bb", "x", job_id="OLD"),
               {"fact_id": "cc", "job_id": "job-1", "eval_found": False}]
    assert feval.labeled_fact_ids(results, job_id="job-1") == {"aa"}
    assert feval.labeled_fact_ids(results, job_id=None) == {"aa", "bb"}


# --- Task 4: OkahuFilteredEval construction + submit -------------------------

def test_submit_requires_exactly_one_template_selector():
    c = _client()
    with pytest.raises(ValueError):
        c.submit("wf", fact_name="traces", start_time="s", end_time="e")
    with pytest.raises(ValueError):
        c.submit("wf", eval_name="hallucination", template={"name": "x"},
                 fact_name="traces", start_time="s", end_time="e")


def test_submit_builtin_sends_template_name_and_no_trace_id():
    c = _client()
    with patch.object(feval.requests, "post",
                      return_value=_resp({"job_id": "job-1", "result": None})) as mock_post:
        job_id = c.submit("wf", eval_name="hallucination", fact_name="traces",
                          start_time="s", end_time="e")
    assert job_id == "job-1"
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"template_name": "hallucination"}
    assert "trace_id" not in kwargs["params"]                  # absence -> filtered mode
    assert kwargs["params"]["workflow_name"] == "wf"


def test_submit_custom_sends_template_object():
    c = _client()
    tmpl = {"name": "hallucination_test", "eval_prompt": "p",
            "structure_output": {"label": {"description": "d"}}}
    with patch.object(feval.requests, "post",
                      return_value=_resp({"job_id": "job-2", "result": None})) as mock_post:
        job_id = c.submit("wf", template=tmpl, fact_name="traces", start_time="s", end_time="e")
    assert job_id == "job-2"
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"template": tmpl}
    assert "trace_id" not in kwargs["params"]


# --- Task 5: poll / fact-id discovery / query / coverage --------------------

def test_query_results_builtin_sends_eval_name_and_bare_hex_ids():
    c = _client()
    payload = {"results": [{"fact_id": "aa", "eval_found": True,
                            "eval_result": {"label": "x"}, "job_id": "j"}]}
    with patch.object(feval.requests, "post", return_value=_resp(payload)) as mock_post:
        rows = c.query_results("wf", ["0xaa"], eval_name="hallucination",
                               fact_name="traces", start_time="s", end_time="e")
    assert len(rows) == 1
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["eval_names"] == ["hallucination"]
    assert kwargs["json"]["fact_ids"] == ["aa"]                # 0x stripped


def test_query_results_custom_sends_custom_and_filters_by_name():
    c = _client()
    payload = {"results": [
        {"fact_id": "aa", "eval_name": "hallucination_test",
         "eval_found": True, "eval_result": {"label": "x"}, "job_id": "j"},
        {"fact_id": "aa", "eval_name": "other_custom",
         "eval_found": True, "eval_result": {"label": "y"}, "job_id": "j"}]}
    with patch.object(feval.requests, "post", return_value=_resp(payload)) as mock_post:
        rows = c.query_results("wf", ["aa"], custom_name="hallucination_test",
                               fact_name="traces", start_time="s", end_time="e")
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["eval_names"] == ["custom"]
    assert [r["eval_result"]["label"] for r in rows] == ["x"]   # other_custom filtered out


def test_get_job_result_fact_ids_returns_bare_hex():
    c = _client()
    payload = {"status": "SUCCEEDED", "results": [{"fact_id": "0xaa"}, {"fact_id": "bb"}]}
    with patch.object(feval.requests, "get", return_value=_resp(payload)):
        assert sorted(c.get_job_result_fact_ids("job-1")) == ["aa", "bb"]


def test_poll_status_returns_terminal_state():
    c = _client()
    with patch.object(feval.requests, "get", return_value=_resp({"status": "SUCCEEDED"})):
        assert c.poll_status("job-1") == "SUCCEEDED"


def test_poll_status_raises_on_failed():
    c = _client()
    with patch.object(feval.requests, "get", return_value=_resp({"status": "FAILED"})):
        with pytest.raises(AssertionError):
            c.poll_status("job-1")


def test_poll_for_coverage_returns_when_all_labeled():
    c = _client()
    rows = [_row("aa", "no_hallucination", job_id="job-1")]
    with patch.object(c, "query_results", return_value=rows) as mock_q:
        latest = c.poll_for_coverage("traces", {"wf": {"aa": True}}, "s", "e",
                                     timeout=30, interval=0, stall_rounds=2, job_id="job-1",
                                     eval_name="hallucination")
    assert latest["wf"] == rows
    assert mock_q.call_count == 1                               # full coverage first round


# --- Task 5b: pagination (get_job_result_fact_ids + query_results) ----------

def _page_resp(rows):
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = {"results": rows}
    return r


def test_query_results_accumulates_across_pages():
    c = feval.OkahuFilteredEval(api_key="k", eval_base="http://e", api_base="http://a")
    page1 = [{"fact_id": f"{i:02x}", "eval_name": "hallucination"} for i in range(100)]
    page2 = [{"fact_id": "aa", "eval_name": "hallucination"}]
    with patch("monocle_test_tools.evals.okahu_filtered_eval.requests.post",
               side_effect=[_page_resp(page1), _page_resp(page2)]) as post:
        rows = c.query_results("wf", [f"{i:02x}" for i in range(101)],
                               eval_name="hallucination", fact_name="traces",
                               start_time="s", end_time="e")
    assert len(rows) == 101
    assert post.call_count == 2


def test_query_results_single_short_page_stops():
    c = feval.OkahuFilteredEval(api_key="k", eval_base="http://e", api_base="http://a")
    with patch("monocle_test_tools.evals.okahu_filtered_eval.requests.post",
               side_effect=[_page_resp([{"fact_id": "aa", "eval_name": "hallucination"}])]) as post:
        rows = c.query_results("wf", ["aa"], eval_name="hallucination",
                               fact_name="traces", start_time="s", end_time="e")
    assert len(rows) == 1
    assert post.call_count == 1


def test_get_job_result_fact_ids_accumulates_across_pages():
    c = feval.OkahuFilteredEval(api_key="k", eval_base="http://e", api_base="http://a")
    page1 = [{"fact_id": f"{i:02x}"} for i in range(100)]
    page2 = [{"fact_id": "ff"}]
    with patch("monocle_test_tools.evals.okahu_filtered_eval.requests.get",
               side_effect=[_page_resp(page1), _page_resp(page2)]) as get:
        ids = c.get_job_result_fact_ids("job-1")
    assert len(ids) == 101
    assert get.call_count == 2


# --- Task 6: run_filtered orchestration -------------------------------------

def test_run_filtered_grades_all_and_enforces_min_facts():
    c = _client()
    coverage = {"wf": [_row("aa", "no_hallucination"), _row("bb", "major_hallucination")]}
    with patch.object(c, "submit", return_value="job-1"), \
         patch.object(c, "poll_status", return_value="SUCCEEDED"), \
         patch.object(c, "get_job_result_fact_ids", return_value=["aa", "bb"]), \
         patch.object(c, "poll_for_coverage", return_value=coverage):
        report = c.run_filtered("wf", accepted="no_hallucination", eval_name="hallucination",
                                fact_name="traces", start_time="s", end_time="e", min_facts=1)
    assert {s["fact_id"]: s["status"] for s in report["scenarios"]} == {"aa": "pass", "bb": "fail"}
    assert report["job_id"] == "job-1"


def test_run_filtered_min_facts_guard_raises_on_too_few():
    c = _client()
    with patch.object(c, "submit", return_value="job-2"), \
         patch.object(c, "poll_status", return_value="SUCCEEDED"), \
         patch.object(c, "get_job_result_fact_ids", return_value=[]):
        with pytest.raises(AssertionError):
            c.run_filtered("wf", accepted="no_hallucination", eval_name="hallucination",
                           fact_name="traces", start_time="s", end_time="e", min_facts=1)
