"""Filtered-eval orchestration for the Okahu eval API (async /v1/eval/jobs jobs).

A "filtered" eval submits one async job scoped by a filter (workflow + time window
+ fact level; no trace_id), lets the server discover every matching fact, and grades
each against a single blanket expectation. Module-scope functions are pure (no I/O)
and unit-tested. Network/env operations live on OkahuFilteredEval. Supports built-in
template_name and inline custom templates.

Only `requests` + the standard library are used here — the filtered-eval path is
deliberately independent of the heavier (bert-score/transformers) evaluators.
"""


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


_COUNT_KEY = {"pass": "passed", "drift": "drift", "error": "errors"}


def _as_list(x):
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def grade_label(actual, accepted=None, not_expected=None) -> str:
    """'pass' | 'drift' | 'missing' for one actual label (any-of accepted + not_expected)."""
    if actual is None:
        return "missing"
    acc, ne = _as_list(accepted), (_as_list(not_expected) or [])
    if acc is not None and actual not in acc:
        return "drift"
    if actual in ne:
        return "drift"
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
    counts = {"passed": 0, "drift": 0, "errors": 0}
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
