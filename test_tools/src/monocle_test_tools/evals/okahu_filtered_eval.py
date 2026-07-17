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
