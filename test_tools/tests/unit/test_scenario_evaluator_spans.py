"""The judge is shown the turn's tool calls."""
from monocle_test_tools.harness.evaluator import ResponseEvaluator, render_tool_calls


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


def test_render_tool_calls_lists_only_tool_invocations():
    spans = [FakeSpan("inference", "gpt-4.1", "prompt", "reply"),
             FakeSpan("agentic.tool.invocation", "book_flight", "Bombay->Hyderabad", "MNCL123")]
    assert render_tool_calls(spans) == "- book_flight(input=Bombay->Hyderabad) -> MNCL123"


def test_render_tool_calls_is_empty_without_tools():
    assert render_tool_calls([FakeSpan("inference", "gpt")]) == ""


def test_judge_prompt_carries_the_tool_calls_block():
    evaluator = ResponseEvaluator(model=None, success_criteria="The flight is booked.")
    prompt = evaluator.build_judge_prompt("Booked!", "turn 1 user> hi", "- book_flight(input=x) -> ok")
    assert "[TOOL CALLS THIS TURN]" in prompt
    assert prompt.index("[LATEST TARGET AGENT RESPONSE]") < prompt.index("[TOOL CALLS THIS TURN]")
    assert "- book_flight(input=x) -> ok" in prompt


def test_judge_prompt_marks_an_empty_tool_call_list():
    evaluator = ResponseEvaluator(model=None, success_criteria="c")
    assert "[TOOL CALLS THIS TURN]\n(none)" in evaluator.build_judge_prompt("r", tool_calls="")


def test_judge_prompt_omits_the_section_when_no_spans_were_given():
    """The tester's self-check tool has no spans; it must not be told 'no tools were called'."""
    evaluator = ResponseEvaluator(model=None, success_criteria="c")
    assert "[TOOL CALLS THIS TURN]" not in evaluator.build_judge_prompt("r")
