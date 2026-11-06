import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.trace import SpanContext, TraceFlags

from monocle_test_tools.evals.eval_manager import get_evaluator
from monocle_test_tools.evals.monocle import MonocleEval
from monocle_test_tools.evals.monocle import evaluator as evaluator_module
from monocle_test_tools.evals.monocle.judge import RunResult
from monocle_test_tools.file_span_loader import JSONSpanLoader
from monocle_test_tools.fluent_api import TraceAssertion

FIXTURES = Path(__file__).parent / "fixtures"
REFUND_TRACE = Path(__file__).parents[2] / "integration" / "json_loader_traces" / "642dbd9d0dfcfdbdc8849f67f34c8a19.json"
TEMPLATE_PATH = Path(__file__).parents[2] / "integration" / "custom_templates" / "refund_confirmation_grounding.json"
FIXTURE_TRACE_ID = "642dbd9d0dfcfdbdc8849f67f34c8a19"
FIXTURE_WORKFLOW = "test_cc_customer_care_agent"
BARE_NAME_MESSAGE = "'monocle' runs custom templates only; pass a template file path"
ONE_SECOND_IN_NANOSECONDS = 1_000_000_000


class FakeJudge:
    def __init__(self, label: str, explanation: str = "because", tokens: int = 10):
        self.label = label
        self.explanation = explanation
        self.tokens = tokens
        self.judged_trace_ids = []

    def judge(self, template, evidence) -> RunResult:
        self.judged_trace_ids.append(evidence.trace_id)
        return RunResult(
            label=self.label,
            explanation=self.explanation,
            judge_output={"label": self.label, "explanation": self.explanation},
            total_tokens=self.tokens,
        )

    def judge_for(self, llm) -> "FakeJudge":
        return self


@pytest.fixture(autouse=True)
def reset_trace_assertion_class_state():
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None


def use_fake_judge(monkeypatch, fake_judge: FakeJudge) -> None:
    monkeypatch.setattr(evaluator_module.judge, "judge_for", fake_judge.judge_for)


def fixture_spans() -> list[ReadableSpan]:
    fixture = json.loads(REFUND_TRACE.read_text())
    return JSONSpanLoader.from_json_str(json.dumps(fixture["spans"]))


def template_dict() -> dict:
    return json.loads(TEMPLATE_PATH.read_text())


def trace_span(trace_id: int, span_id: int, question: str, answer: str) -> ReadableSpan:
    context = SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False, trace_flags=TraceFlags(1))
    return ReadableSpan(
        name="turn",
        context=context,
        attributes={"span.type": "agentic.turn", "workflow.name": "two_traces"},
        events=[
            Event("data.input", {"input": question}, timestamp=span_id * ONE_SECOND_IN_NANOSECONDS),
            Event("data.output", {"response": answer}, timestamp=span_id * ONE_SECOND_IN_NANOSECONDS + 1),
        ],
        start_time=span_id * ONE_SECOND_IN_NANOSECONDS,
        end_time=span_id * ONE_SECOND_IN_NANOSECONDS + 2,
    )


def test_get_evaluator_returns_monocle_eval():
    evaluator = get_evaluator("monocle", {"trace_source": "", "repeat": 3, "min_agreement": 0.6})

    assert isinstance(evaluator, MonocleEval)
    assert evaluator.eval_options == {"trace_source": "", "repeat": 3, "min_agreement": 0.6}


def test_unknown_option_key_rejected():
    with pytest.raises(ValueError, match="temperature"):
        MonocleEval(eval_options={"trace_source": "", "temperature": 0.5})


@pytest.mark.parametrize("min_agreement", [0, -0.5, 1.2, "high", True])
def test_min_agreement_outside_zero_to_one_rejected(min_agreement):
    with pytest.raises(ValueError, match="min_agreement must be a number above 0 and at most 1"):
        MonocleEval(eval_options={"min_agreement": min_agreement})


@pytest.mark.parametrize("min_agreement", [0.5, 1])
def test_min_agreement_inside_zero_to_one_accepted(min_agreement):
    assert MonocleEval(eval_options={"min_agreement": min_agreement}).eval_options["min_agreement"] == min_agreement


def test_bare_template_name_rejected():
    evaluator = MonocleEval(eval_options={})

    with pytest.raises(ValueError) as raised:
        evaluator.evaluate(filtered_spans=fixture_spans(), eval_name="hallucination", template=None)

    assert str(raised.value) == BARE_NAME_MESSAGE


def test_any_fact_name_but_traces_is_not_supported(monkeypatch):
    use_fake_judge(monkeypatch, FakeJudge("grounded"))
    evaluator = MonocleEval(eval_options={})

    with pytest.raises(ValueError) as raised:
        evaluator.evaluate(filtered_spans=fixture_spans(), fact_name="something_else", template=template_dict())

    assert str(raised.value) == 'with_evaluation("monocle") judges traces; fact_name="something_else" is not supported'


def test_no_spans_fails(monkeypatch):
    use_fake_judge(monkeypatch, FakeJudge("grounded"))

    with pytest.raises(AssertionError, match="No spans to judge"):
        MonocleEval(eval_options={}).evaluate(filtered_spans=[], template=template_dict())


def test_evaluate_returns_label_and_fills_fact_results(monkeypatch):
    use_fake_judge(monkeypatch, FakeJudge("ungrounded", explanation="invented refund id", tokens=42))
    evaluator = MonocleEval(eval_options={})

    label, explanation = evaluator.evaluate(filtered_spans=fixture_spans(), template=template_dict())

    assert (label, explanation) == ("ungrounded", "invented refund id")
    assert evaluator.last_fact_results == [{
        "fact_id": FIXTURE_TRACE_ID,
        "job_id": None,
        "eval_found": True,
        "eval_result": {"label": "ungrounded", "explanation": "invented refund id"},
        "workflow": FIXTURE_WORKFLOW,
        "judge": {"label": "ungrounded", "runs": 1, "agreement": 1.0, "confidence": None,
                  "probabilities": None, "inconclusive": None},
    }]
    assert evaluator.last_judge_description == "openai:gpt-4.1-mini, 1 run per trace"
    assert evaluator.last_judge_output == {"label": "ungrounded", "explanation": "invented refund id"}
    assert evaluator.last_total_tokens == 42


def test_each_trace_is_judged_separately(monkeypatch):
    fake_judge = FakeJudge("grounded", tokens=5)
    use_fake_judge(monkeypatch, fake_judge)
    spans = [trace_span(0xA1, 1, "refund?", "done"), trace_span(0xA2, 2, "when?", "soon")]
    evaluator = MonocleEval(eval_options={})

    evaluator.evaluate(filtered_spans=spans, template=template_dict())

    assert fake_judge.judged_trace_ids == [f"{0xA1:032x}", f"{0xA2:032x}"]
    assert [result["fact_id"] for result in evaluator.last_fact_results] == [f"{0xA1:032x}", f"{0xA2:032x}"]
    assert evaluator.last_total_tokens == 10


def test_with_evaluation_repeat_reaches_evaluator(monkeypatch):
    fake_judge = FakeJudge("grounded")
    use_fake_judge(monkeypatch, fake_judge)
    asserter = TraceAssertion(filtered_spans=fixture_spans())

    asserter.with_evaluation("monocle", repeat=3)
    asserter._eval.evaluate(filtered_spans=fixture_spans(), template=template_dict())

    assert asserter._eval.eval_options["repeat"] == 3
    assert len(fake_judge.judged_trace_ids) == 3


def test_with_evaluation_rejects_repeat_in_eval_options():
    asserter = TraceAssertion(filtered_spans=fixture_spans())

    with pytest.raises(ValueError) as raised:
        asserter.with_evaluation("monocle", {"repeat": 5})

    assert str(raised.value) == "pass repeat= to with_evaluation as a keyword, not in eval_options"


def test_with_evaluation_rejects_repeat_below_one():
    asserter = TraceAssertion(filtered_spans=fixture_spans())

    with pytest.raises(ValueError, match="repeat must be at least 1, got 0"):
        asserter.with_evaluation("monocle", repeat=0)


def test_check_eval_end_to_end_with_fake_judge(monkeypatch):
    use_fake_judge(monkeypatch, FakeJudge("ungrounded", explanation="invented refund id"))
    asserter = TraceAssertion(filtered_spans=fixture_spans())
    asserter.with_evaluation("monocle")

    try:
        passing = asserter.check_eval(str(TEMPLATE_PATH), expected="ungrounded")
        assert not passing.has_assertions()

        failing = asserter.check_eval(str(TEMPLATE_PATH), expected="grounded")
        assert failing.has_assertions()
        message = failing.get_assertion_messages()
        assert FIXTURE_TRACE_ID in message
        assert "expected   grounded" in message
        assert f"{FIXTURE_TRACE_ID}  →  ungrounded" in message
        assert "invented refund id" in message
    finally:
        TraceAssertion._assertion_errors = []


def test_several_failing_traces_are_reported_under_the_template_name(monkeypatch):
    use_fake_judge(monkeypatch, FakeJudge("ungrounded"))
    spans = [trace_span(0xA1, 1, "refund?", "done"), trace_span(0xA2, 2, "when?", "soon")]
    asserter = TraceAssertion(filtered_spans=spans)
    asserter.with_evaluation("monocle")

    try:
        failing = asserter.check_eval(str(TEMPLATE_PATH), expected="grounded")
        message = failing.get_assertion_messages()
        assert "Eval 'refund_confirmation_grounding': 2 of 2 did not match" in message
        assert f"{0xA1:032x}" in message and f"{0xA2:032x}" in message
    finally:
        TraceAssertion._assertion_errors = []


def test_monocle_eval_with_time_window_source_is_rejected():
    window = {"workflows": ["refunds"], "start_time": "2026-10-01T00:00:00Z", "end_time": "2026-10-02T00:00:00Z"}
    asserter = TraceAssertion(okahu_filter=window)
    asserter.with_evaluation("monocle")

    with pytest.raises(ValueError) as raised:
        asserter.check_eval(str(TEMPLATE_PATH), expected="ungrounded")

    assert str(raised.value) == (
        "time-window evals run on Okahu only; use with_evaluation('okahu') or a single trace source"
    )


def test_repeat_with_okahu_evaluator_is_rejected():
    asserter = TraceAssertion(filtered_spans=fixture_spans())

    with pytest.raises(ValueError) as raised:
        asserter.with_evaluation("okahu", repeat=3)

    assert str(raised.value) == "repeat is supported by with_evaluation('monocle') only"


def test_okahu_evaluator_accepts_the_default_repeat():
    asserter = TraceAssertion(filtered_spans=fixture_spans())

    asserter.with_evaluation("okahu")

    assert asserter._eval.eval_options["repeat"] == 1
