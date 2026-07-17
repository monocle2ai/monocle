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


# --- Task 2: blanket grading + filtered report assembly (job_id-gated) --------

def _row(fid, label, job_id="job-1", finish_type=None, workflow="wf"):
    return {"fact_id": fid, "job_id": job_id, "eval_found": True, "workflow": workflow,
            "eval_result": {"label": label, "explanation": "why"},
            "finish_type": finish_type}


def test_grade_label_any_of_and_not_expected():
    assert feval.grade_label("minor_hallucination", accepted=["minor_hallucination", "no_hallucination"]) == "pass"
    assert feval.grade_label("major_hallucination", accepted=["no_hallucination"]) == "drift"
    assert feval.grade_label("major_hallucination", not_expected=["minor_hallucination", "major_hallucination"]) == "drift"
    assert feval.grade_label("no_hallucination", not_expected=["major_hallucination"]) == "pass"
    assert feval.grade_label(None, accepted=["no_hallucination"]) == "missing"


def test_filtered_report_grades_every_returned_fact_and_gates_on_job_id():
    results = [_row("aa", "no_hallucination", job_id="job-1"),
               _row("bb", "major_hallucination", job_id="job-1"),
               _row("cc", "no_hallucination", job_id="OLD")]        # stale -> gated out entirely
    report = feval.build_filtered_report("no_hallucination", None, results, job_id="job-1")
    by_fid = {s["fact_id"]: s["status"] for s in report["scenarios"]}
    assert by_fid == {"aa": "pass", "bb": "drift"}                  # cc excluded (wrong job)
    assert report["summary"] == {"total": 2, "passed": 1, "drift": 1,
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
    assert report["scenarios"][0]["status"] == "drift"


# --- Task 3: labeled-coverage set (poller support) ---------------------------

def test_labeled_fact_ids_gated_by_job_id():
    results = [_row("aa", "x", job_id="job-1"),
               _row("bb", "x", job_id="OLD"),
               {"fact_id": "cc", "job_id": "job-1", "eval_found": False}]
    assert feval.labeled_fact_ids(results, job_id="job-1") == {"aa"}
    assert feval.labeled_fact_ids(results, job_id=None) == {"aa", "bb"}
