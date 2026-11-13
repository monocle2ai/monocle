import pytest

from monocle_test_tools.evals.monocle.judge import RunResult
from monocle_test_tools.evals.monocle.results import EvalResult, row_for_check_eval, settle
from monocle_test_tools.evals.monocle.template import EvalTemplate, EvidenceScope, LlmConfig, OutputField

LABELS = ["grounded", "partially_grounded", "ungrounded"]


def template_for(llm: LlmConfig) -> EvalTemplate:
    return EvalTemplate(
        name="grounding",
        eval_prompt="Is the answer grounded?",
        structure_output={
            "label": OutputField(description="verdict", enums=LABELS),
            "explanation": OutputField(description="why", type="string"),
        },
        evidence_scope=EvidenceScope.default(),
        llm=llm,
    )


LLM_TEMPLATE = template_for(LlmConfig(model="openai:gpt-4.1"))
DECISION_TEMPLATE = template_for(LlmConfig(model="typesafe:jev-latest", min_confidence=0.8))
TRACE_ID = "trace-1"


def run_of(label: str, tokens: int = 0, explanation: str = "because") -> RunResult:
    return RunResult(label=label, explanation=explanation, judge_output={"label": label, "explanation": explanation},
                     total_tokens=tokens)


def decision_run_of(label: str, probabilities: dict[str, float]) -> RunResult:
    return RunResult(label=label, explanation="jev", judge_output={}, probabilities=probabilities, total_tokens=0)


def test_single_run_is_its_own_verdict():
    verdict = settle([run_of("grounded")], LLM_TEMPLATE, TRACE_ID)
    assert verdict.label == "grounded"
    assert verdict.trace_id == TRACE_ID
    assert verdict.agreement == 1.0
    assert verdict.confidence is None


def test_majority_label_wins():
    runs = [run_of("grounded"), run_of("ungrounded"), run_of("grounded")]
    verdict = settle(runs, LLM_TEMPLATE, TRACE_ID, min_agreement=0.6)
    assert verdict.label == "grounded"
    assert verdict.agreement == pytest.approx(2 / 3)


def test_below_min_agreement_is_inconclusive_even_when_majority_matches():
    runs = [run_of("grounded")] * 3 + [run_of("ungrounded")] * 2
    verdict = settle(runs, LLM_TEMPLATE, TRACE_ID)
    assert verdict.label == "grounded"
    assert verdict.inconclusive == "flipped: grounded ×3, ungrounded ×2 (60% agree, need 80%)"


def test_a_conclusive_verdict_has_no_reason():
    runs = [run_of("ungrounded")] * 4 + [run_of("grounded")]
    assert settle(runs, LLM_TEMPLATE, TRACE_ID).inconclusive is None


def test_verdict_sums_tokens_and_keeps_the_first_majority_run_output():
    runs = [run_of("ungrounded", 5, "first"), run_of("grounded", 7, "second"), run_of("grounded", 11, "third")]
    verdict = settle(runs, LLM_TEMPLATE, TRACE_ID, min_agreement=0.6)
    assert verdict.total_tokens == 23
    assert verdict.explanation == "second"
    assert verdict.judge_output["explanation"] == "second"
    assert len(verdict.runs) == 3


def test_decision_confidence_is_the_probability_of_the_chosen_label():
    run = decision_run_of("grounded", {"grounded": 0.85, "partially_grounded": 0.1, "ungrounded": 0.05})
    verdict = settle([run], DECISION_TEMPLATE, TRACE_ID)
    assert verdict.confidence == 0.85


def test_decision_below_min_confidence_is_inconclusive():
    run = decision_run_of("ungrounded", {"grounded": 0.07, "partially_grounded": 0.31, "ungrounded": 0.62})
    verdict = settle([run], DECISION_TEMPLATE, TRACE_ID)
    assert verdict.label == "ungrounded"
    assert verdict.confidence == 0.62
    assert verdict.inconclusive == "low confidence: ungrounded 0.62 (need 0.80)"


def test_decision_without_min_confidence_accepts_any_choice():
    template = template_for(LlmConfig(model="typesafe:jev-latest"))
    run = decision_run_of("grounded", {"grounded": 0.4, "ungrounded": 0.35, "partially_grounded": 0.25})
    assert settle([run], template, TRACE_ID).confidence == 0.4


def test_row_for_check_eval_has_the_shape_check_eval_grades_plus_the_judge_numbers():
    verdict = EvalResult(trace_id=TRACE_ID, label="grounded", explanation="because", judge_output={},
                         runs=[run_of("grounded")] * 3, agreement=1.0, confidence=None, total_tokens=3)
    assert row_for_check_eval(verdict, "my_workflow") == {
        "fact_id": TRACE_ID,
        "job_id": None,
        "eval_found": True,
        "eval_result": {"label": "grounded", "explanation": "because"},
        "workflow": "my_workflow",
        "judge": {"label": "grounded", "runs": 3, "agreement": 1.0, "confidence": None,
                  "probabilities": None, "inconclusive": None},
    }


def test_an_inconclusive_row_is_labelled_inconclusive_so_nothing_matches_it():
    runs = [run_of("grounded")] * 3 + [run_of("ungrounded")] * 2
    row = row_for_check_eval(settle(runs, LLM_TEMPLATE, TRACE_ID), "wf")
    assert row["eval_result"]["label"] == "inconclusive"
    assert row["judge"]["label"] == "grounded"
    assert row["judge"]["inconclusive"].startswith("flipped")
