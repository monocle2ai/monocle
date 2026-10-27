"""The judge is shown the criteria, the conversation and the agent's tool calls, so a claimed booking needs a call."""
from monocle_test_tools.harness.evaluator import ResponseEvaluator, render_tool_calls


class Span:
    def __init__(self, span_type, name, inputs=None, output=None):
        self.attributes = {"span.type": span_type, "entity.1.name": name}
        self.events = [type("E", (), {"name": "data.input", "attributes": {"input": inputs}})(),
                       type("E", (), {"name": "data.output", "attributes": {"response": output}})()]


def test_only_tool_calls_are_listed_for_the_judge():
    spans = [Span("inference", "gpt-4.1", "prompt", "reply"),
             Span("agentic.tool.invocation", "book_flight", "Bombay->Hyderabad", "MNCL123")]
    assert render_tool_calls(spans) == "- book_flight(input=Bombay->Hyderabad) -> MNCL123"


def test_the_prompt_has_criteria_conversation_reply_then_tool_calls():
    prompt = ResponseEvaluator(None, "Flight is booked").build_judge_prompt("Booked!", "turn 1 user> hi", "")
    for section in ("Flight is booked", "turn 1 user> hi", "[LATEST TARGET AGENT RESPONSE]\nBooked!",
                    "[TOOL CALLS THIS TURN]\n(none)"):
        assert section in prompt
    assert prompt.index("Booked!") < prompt.index("[TOOL CALLS THIS TURN]")
