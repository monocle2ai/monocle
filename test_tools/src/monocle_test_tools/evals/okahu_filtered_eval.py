"""Filtered-eval orchestration for the Okahu eval API (async /v1/eval/jobs jobs).

A "filtered" eval submits one async job scoped by a filter (workflow + time window
+ fact level; no trace_id), lets the server discover every matching fact, and grades
each against a single blanket expectation. Module-scope functions are pure (no I/O)
and unit-tested. Network/env operations live on OkahuFilteredEval. Supports built-in
template_name and inline custom templates.

Only `requests` + the standard library are used here — the filtered-eval path is
deliberately independent of the heavier (bert-score/transformers) evaluators.
"""
import os
import time

import requests


def normalize_fact_id(fid) -> str:
    """Bare-hex fact id (strip a leading 0x). Format-agnostic to caller input."""
    s = str(fid)
    return s[2:] if s.startswith("0x") else s


def has_label(result) -> bool:
    """True if an API result row carries a usable eval label."""
    return bool(
        result
        and result.get("eval_found")
        and (result.get("eval_result") or {}).get("label")
    )


_COUNT_KEY = {"pass": "passed", "fail": "failed", "error": "errors"}
_DEFAULT_PAGE_SIZE = 100


def _as_list(x):
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def grade_label(actual, accepted=None, not_expected=None) -> str:
    """'pass' | 'fail' | 'missing' for one actual label (any-of accepted + not_expected)."""
    if actual is None:
        return "missing"
    acc, ne = _as_list(accepted), (_as_list(not_expected) or [])
    if acc is not None and actual not in acc:
        return "fail"
    if actual in ne:
        return "fail"
    return "pass"


def _index_by_fact(results, job_id):
    """Bare-hex fact_id -> row, keeping only rows produced by job_id (when given)."""
    by_fid = {}
    for r in results:
        if not r:
            continue
        if job_id is not None and r.get("job_id") != job_id:
            continue
        by_fid[normalize_fact_id(r.get("fact_id", ""))] = r
    return by_fid


def _grade_row(row, accepted, not_expected):
    """(status, actual, explanation) for a single job-gated row."""
    if row is None:
        return "missing", None, ""
    if row.get("finish_type") == "error":
        return "error", None, (row.get("finish_details") or "")
    if not has_label(row):
        return "missing", None, ""
    actual = (row.get("eval_result") or {}).get("label")
    explanation = " ".join(((row.get("eval_result") or {}).get("explanation") or "").split())
    return grade_label(actual, accepted, not_expected), actual, explanation


def build_filtered_report(accepted, not_expected, results: list, job_id,
                          duration_seconds=None) -> dict:
    """Grade EVERY job-gated fact against a blanket accepted/not_expected.

    There is no known id list, so there is no "missing"; the min_facts guard (enforced by
    run_filtered) protects against a vacuous pass on too few facts.
    """
    exp_label = _as_list(accepted) or ["not:" + str(_as_list(not_expected))]
    scenarios = []
    for fid, row in _index_by_fact(results, job_id).items():
        status, actual, explanation = _grade_row(row, accepted=accepted, not_expected=not_expected)
        if status == "missing":   # discovered rows always carry a label or an error
            continue
        scenarios.append({
            "fact_id": fid, "expected": exp_label, "actual": actual, "status": status,
            "job_id": row.get("job_id"), "explanation": explanation,
            "workflow": row.get("workflow", ""),
        })
    counts = {"passed": 0, "failed": 0, "errors": 0}
    for s in scenarios:
        counts[_COUNT_KEY[s["status"]]] += 1
    return {
        "job_id": job_id, "status": "completed",
        "summary": {"total": len(scenarios), **counts, "duration_seconds": duration_seconds},
        "scenarios": scenarios,
    }


def labeled_fact_ids(results: list, job_id) -> set:
    """Bare-hex ids with a usable label produced by job_id (gate off when job_id is None)."""
    return {
        normalize_fact_id(r.get("fact_id", ""))
        for r in results
        if has_label(r) and (job_id is None or r.get("job_id") == job_id)
    }


class OkahuFilteredEval:
    """Env-bound HTTP client for the Okahu filtered-eval (async job) + query APIs.

    A filtered eval is submitted with NO trace_id, so the server runs an async job that
    discovers every fact matching the filter. Supports built-in template_name and inline
    custom templates. Ported from the proven evaluation-testing BatchEvalClient, plus the
    custom-template switch.
    """

    def __init__(self, api_key, eval_base, api_base, poll_interval_s=5, poll_timeout_s=600):
        self.headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        self.eval_base = eval_base.rstrip("/")
        self.api_base = api_base.rstrip("/") + "/api"
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s

    @classmethod
    def from_env(cls):
        required = ["OKAHU_API_KEY", "OKAHU_EVALUATION_ENDPOINT", "OKAHU_API_ENDPOINT"]
        for var in required:
            if not os.environ.get(var):
                raise RuntimeError(f"Missing required environment variable: {var}")
        return cls(
            api_key=os.environ["OKAHU_API_KEY"],
            eval_base=os.environ["OKAHU_EVALUATION_ENDPOINT"],
            api_base=os.environ["OKAHU_API_ENDPOINT"],
            poll_interval_s=int(os.getenv("BATCH_EVAL_JOB_POLL_INTERVAL_S", "5")),
            poll_timeout_s=int(os.getenv("BATCH_EVAL_JOB_POLL_TIMEOUT_S", "600")),
        )

    def submit(self, workflow_names, *, eval_name=None, template=None,
               fact_name, start_time, end_time) -> str:
        """Submit a filtered eval job (no trace_id) and return its job_id.

        Exactly one of `eval_name` (built-in template) or `template` (inline custom).
        """
        if bool(eval_name) == bool(template):
            raise ValueError("Provide exactly one of 'eval_name' or 'template'.")
        url = f"{self.eval_base}/v1/eval/jobs"
        params = {
            "workflow_name": workflow_names,
            "start_time": start_time, "end_time": end_time,
            "fact_name": fact_name, "breakdown_filter": fact_name,
            "shadow_eval": False,
        }  # NOTE: no trace_id -> async filtered mode. shadow_eval is False because
        # filtered jobs grade traces already loaded in Okahu, not test-generated
        # shadow traces (mirrors the interactive path's `_trace_source != "okahu"`).
        body = {"template": template} if template else {"template_name": eval_name}
        resp = requests.post(url, headers=self.headers, params=params, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("job_id")
        assert job_id, f"No job_id in response: {data}"
        assert data.get("result") is None, f"Expected filtered mode (result=None), got: {data}"
        return job_id

    def poll_status(self, job_id) -> str:
        """Block until the job reaches a terminal state; return it (raise on failure/timeout)."""
        url = f"{self.eval_base}/v1/eval/jobs/{job_id}/status"
        deadline = time.time() + self.poll_timeout_s
        while time.time() < deadline:
            resp = requests.get(url, headers=self.headers, timeout=60)
            resp.raise_for_status()
            status = resp.json().get("status", "").upper()
            if status in ("SUCCEEDED", "COMPLETED", "FINISHED"):
                return status
            if status in ("FAILED", "ERROR"):
                raise AssertionError(f"Filtered eval job {job_id} failed with status: {status}")
            time.sleep(self.poll_interval_s)
        raise AssertionError(f"Filtered eval job {job_id} timed out after {self.poll_timeout_s}s")

    def _paginate_get(self, url):
        """Yield every 'results' row across pages for a GET endpoint (limit/offset)."""
        offset = 0
        while True:
            resp = requests.get(url, headers=self.headers,
                                params={"limit": _DEFAULT_PAGE_SIZE, "offset": offset},
                                timeout=60)
            resp.raise_for_status()
            rows = resp.json().get("results", []) or []
            for r in rows:
                yield r
            if len(rows) < _DEFAULT_PAGE_SIZE:
                return
            offset += _DEFAULT_PAGE_SIZE

    def _paginate_post(self, url, body):
        """Yield every 'results' row across pages for a POST query (limit/offset in body)."""
        offset = 0
        while True:
            page_body = {**body, "limit": _DEFAULT_PAGE_SIZE, "offset": offset}
            resp = requests.post(url, headers=self.headers, json=page_body, timeout=60)
            resp.raise_for_status()
            rows = resp.json().get("results", []) or []
            for r in rows:
                yield r
            if len(rows) < _DEFAULT_PAGE_SIZE:
                return
            offset += _DEFAULT_PAGE_SIZE

    def get_job_result_fact_ids(self, job_id) -> list:
        """Bare-hex fact ids this job evaluated (from the job-detail results array).

        VERIFY the row key against a live job before trusting this — the field may be
        'fact_id', 'id', or nested. Adjust the extraction below to match (see plan
        Phase 5 Step 2).
        """
        url = f"{self.eval_base}/v1/eval/jobs/{job_id}"
        ids = [normalize_fact_id(r.get("fact_id"))
               for r in self._paginate_get(url) if r and r.get("fact_id")]
        return sorted(set(ids))

    def query_results(self, workflow_name, fact_ids, *, eval_name=None, custom_name=None,
                      fact_name, start_time, end_time) -> list:
        """Query eval_results for a fact-id set via the canonical /evals/query path.

        Built-in sends eval_names=[eval_name]; custom sends eval_names=["custom"] and filters
        returned rows to eval_name == custom_name (the backend returns each custom row with its
        original name, not the literal "custom"). All ids are sent bare-hex (Lesson 1).
        """
        url = f"{self.api_base}/v1/workflows/{workflow_name}/evals/query"
        eval_names = ["custom"] if custom_name else [eval_name]
        body = {
            "eval_names": eval_names, "fact_name": fact_name,
            "fact_ids": [normalize_fact_id(f) for f in fact_ids],
            "start_time": start_time, "end_time": end_time,
        }
        results = list(self._paginate_post(url, body))
        if custom_name:
            results = [r for r in results if r and r.get("eval_name") == custom_name]
        return results

    def poll_for_coverage(self, fact_name, per_workflow, start_time, end_time, *,
                          timeout, interval, stall_rounds, job_id,
                          eval_name=None, custom_name=None) -> dict:
        """job_id-gated, plateau-aware poller over the discovered fact ids per workflow.

        Stops on full coverage, on the labeled-count plateauing for `stall_rounds` rounds, or
        on hard timeout — returning the latest results seen for each workflow.
        """
        deadline = time.time() + timeout
        latest = {wf: [] for wf in per_workflow}
        best_labeled, stalled = -1, 0
        while True:
            incomplete, total_labeled = {}, 0
            for wf, mapped in per_workflow.items():
                fact_ids = [normalize_fact_id(f) for f in mapped]
                results = self.query_results(
                    wf, fact_ids, eval_name=eval_name, custom_name=custom_name,
                    fact_name=fact_name, start_time=start_time, end_time=end_time)
                latest[wf] = results
                labeled = labeled_fact_ids(results, job_id)
                total_labeled += len(labeled)
                missing = set(fact_ids) - labeled
                if missing:
                    incomplete[wf] = len(missing)
            if not incomplete:
                return latest
            if total_labeled > best_labeled:
                best_labeled, stalled = total_labeled, 0
            else:
                stalled += 1
                if stalled >= stall_rounds:
                    return latest
            if time.time() >= deadline:
                return latest
            time.sleep(interval)

    def run_filtered(self, workflow_names, *, accepted=None, not_expected=None,
                     eval_name=None, template=None, fact_name, start_time, end_time,
                     min_facts=1, max_facts=None, results_timeout=300, results_interval=10,
                     stall_rounds=2) -> dict:
        """Submit -> poll -> enumerate discovered facts -> reconcile -> report.

        Grades EVERY fact the server discovered against the blanket accepted/not_expected.
        Raises if fewer than `min_facts` facts were discovered (never a vacuous pass).
        """
        custom_name = template.get("name") if template else None
        wf_param = workflow_names if isinstance(workflow_names, str) else ",".join(workflow_names)
        workflows = [workflow_names] if isinstance(workflow_names, str) else list(workflow_names)
        job_id = self.submit(wf_param, eval_name=eval_name, template=template,
                             fact_name=fact_name, start_time=start_time, end_time=end_time)
        self.poll_status(job_id)
        discovered = self.get_job_result_fact_ids(job_id)
        if len(discovered) < min_facts:
            raise AssertionError(
                f"Filtered eval evaluated {len(discovered)} fact(s) (min_facts={min_facts}); "
                f"check workflow/time-window filter. job_id={job_id}")
        ceiling = max_facts if max_facts is not None else int(os.getenv("OKAHU_MAX_FACTS", "1000"))
        if len(discovered) > ceiling:
            raise AssertionError(
                f"Filtered eval discovered {len(discovered)} facts, exceeding max_facts={ceiling} "
                f"(set OKAHU_MAX_FACTS to raise the ceiling, or narrow the time window). "
                f"job_id={job_id}")
        # Reconcile via the canonical /evals/query path over the discovered ids (reuses coverage).
        per_workflow = {wf: {fid: True for fid in discovered} for wf in workflows}
        latest = self.poll_for_coverage(
            fact_name, per_workflow, start_time, end_time,
            timeout=results_timeout, interval=results_interval, stall_rounds=stall_rounds,
            job_id=job_id, eval_name=eval_name, custom_name=custom_name)
        # Tag each row with its source workflow (query rows don't carry it) so the
        # report/matrix can attribute a fact to a workflow.
        merged_results = [{**r, "workflow": r.get("workflow") or wf}
                          for wf in workflows for r in latest.get(wf, []) if r]
        return build_filtered_report(accepted, not_expected, merged_results, job_id=job_id)
