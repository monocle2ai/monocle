"""Settles a judge's runs into one verdict per trace, and shapes the verdicts the way check_eval reads them."""
from collections import Counter

from pydantic import BaseModel

from monocle_test_tools.evals.monocle.judge import RunResult
from monocle_test_tools.evals.monocle.template import EvalTemplate


class EvalResult(BaseModel):
    trace_id: str
    label: str
    explanation: str
    judge_output: dict
    runs: list[RunResult]
    agreement: float
    confidence: float | None
    total_tokens: int
    inconclusive: str | None = None

INCONCLUSIVE_LABEL = "inconclusive"


def settle(runs: list[RunResult], template: EvalTemplate, trace_id: str, min_agreement: float = 0.8) -> EvalResult:
    """Return the majority verdict of the runs.

    The verdict is marked inconclusive, with the reason, when fewer than `min_agreement` of the
    runs agree, or when a decision model's probability for its choice is under the template's
    `llm.min_confidence`.
    """
    label_counts = Counter(run.label for run in runs).most_common()
    majority_label, majority_count = label_counts[0]
    agreement = majority_count / len(runs)
    majority_run = next(run for run in runs if run.label == majority_label)
    confidence = majority_run.probabilities.get(majority_label) if majority_run.probabilities else None
    return EvalResult(
        trace_id=trace_id,
        label=majority_label,
        explanation=majority_run.explanation,
        judge_output=majority_run.judge_output,
        runs=runs,
        agreement=agreement,
        confidence=confidence,
        total_tokens=sum(run.total_tokens for run in runs),
        inconclusive=_why_inconclusive(label_counts, agreement, min_agreement, majority_run, confidence, template),
    )


def row_for_check_eval(verdict: EvalResult, workflow_name: str) -> dict:
    """One entry of last_fact_results: the shape check_eval grades, plus the judge's numbers for the report.

    An inconclusive verdict is labelled "inconclusive" so no expectation can match it."""
    label = INCONCLUSIVE_LABEL if verdict.inconclusive else verdict.label
    return {
        "fact_id": verdict.trace_id,
        "job_id": None,
        "eval_found": True,
        "eval_result": {"label": label, "explanation": verdict.explanation},
        "workflow": workflow_name,
        "judge": {
            "label": verdict.label,
            "runs": len(verdict.runs),
            "agreement": verdict.agreement,
            "confidence": verdict.confidence,
            "probabilities": dict(verdict.runs[0].probabilities) if verdict.runs and verdict.runs[0].probabilities else None,
            "inconclusive": verdict.inconclusive,
        },
    }


def _why_inconclusive(label_counts, agreement, min_agreement, majority_run, confidence, template) -> str | None:
    if agreement < min_agreement:
        counts_text = ", ".join(f"{label} ×{count}" for label, count in label_counts)
        return f"flipped: {counts_text} ({agreement:.0%} agree, need {min_agreement:.0%})"
    min_confidence = template.llm.min_confidence
    if confidence is not None and min_confidence is not None and confidence < min_confidence:
        return f"low confidence: {majority_run.label} {confidence:.2f} (need {min_confidence:.2f})"
    return None
