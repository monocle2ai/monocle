"""The judge prompt. judge() itself needs a live model and is covered by the integration test."""
from monocle_test_tools.harness.evaluator import ResponseEvaluator, Verdict, render_tool_calls


class FakeEvent:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = attributes


class FakeSpan:
    def __init__(self, span_type, name, input_text=None, output_text=None):
        self.attributes = {"span.type": span_type, "entity.1.name": name}
        self.events = []
        if input_text is not None:
            self.events.append(FakeEvent("data.input", {"input": input_text}))
        if output_text is not None:
            self.events.append(FakeEvent("data.output", {"response": output_text}))


def _evaluator():
    return ResponseEvaluator(model=None, success_criteria="Flight is booked")


def test_verdict_requires_met_and_reason():
    verdict = Verdict(met=True, reason="booking confirmed")
    assert verdict.met is True and verdict.confidence is None


def test_judge_prompt_carries_criteria_transcript_and_response():
    prompt = _evaluator().build_judge_prompt({"status": "booked"}, "turn 1 tester> hi")
    assert "Flight is booked" in prompt and "turn 1 tester> hi" in prompt and "booked" in prompt
    assert "(none)" in _evaluator().build_judge_prompt("ok", "")


def test_render_tool_calls_lists_only_tool_invocations():
    spans = [FakeSpan("inference", "gpt-4.1", "prompt", "reply"),
             FakeSpan("agentic.tool.invocation", "book_flight", "Bombay->Hyderabad", "MNCL123")]
    assert render_tool_calls(spans) == "- book_flight(input=Bombay->Hyderabad) -> MNCL123"
    assert render_tool_calls([FakeSpan("inference", "gpt")]) == ""


def test_judge_prompt_shows_tool_calls_after_the_response_when_given():
    prompt = _evaluator().build_judge_prompt("Booked!", "", "- book_flight(input=x) -> ok")
    assert prompt.index("[LATEST TARGET AGENT RESPONSE]") < prompt.index("[TOOL CALLS THIS TURN]")
    assert "- book_flight(input=x) -> ok" in prompt
    assert "[TOOL CALLS THIS TURN]\n(none)" in _evaluator().build_judge_prompt("r", tool_calls="")


def test_judge_prompt_omits_the_tool_section_when_no_spans_were_given():
    assert "[TOOL CALLS THIS TURN]" not in _evaluator().build_judge_prompt("r")


def test_verdict_violated_defaults_false_and_the_preamble_defines_it():
    from monocle_test_tools.harness.evaluator import JUDGE_PREAMBLE
    assert Verdict(met=False, reason="r").violated is False
    assert "violated" in JUDGE_PREAMBLE
