"""A failed scenario explains itself: the reason and every turn, with no span noise in the assert message."""
from monocle_test_tools.harness.driver import ScenarioHarness
from monocle_test_tools.harness.evaluator import Verdict
from monocle_test_tools.harness.result import ScenarioResult, TurnRecord
from monocle_test_tools.harness.test_agent import ParamCall

TURNS = [TurnRecord(turn=1, tester_message="I need a flight out of San Francisco.", target_response="Where to?",
                    verdict=Verdict(met=False, reason="no destination yet")),
         TurnRecord(turn=2, tester_message="Seattle, obviously.", target_response="What date?",
                    verdict=Verdict(met=False, reason="not booked"),
                    param_tools_called=[ParamCall(name="destination", value="Seattle")])]


def test_the_report_gives_the_reason_then_each_turn_with_tool_calls_on_their_own_line():
    lines = ScenarioResult(test_name="t", scenario="Book a flight.", passed=False, turns_used=2, max_turns=2,
                           failure_reason="max turns exhausted", turns=TURNS).report().splitlines()
    assert lines[:2] == ["FAILED scenario 'Book a flight.' after 2/2 turns", "  reason: max turns exhausted"]
    assert lines[lines.index("  turn 2  tester> [called destination()] ==> Seattle") + 1] == \
        "          tester> Seattle, obviously."
    assert "          verdict> not met -- not booked" in lines


def test_repr_leaves_out_spans_and_turns():
    class Span:
        def __repr__(self):
            return "<SPAN>"
    result = ScenarioResult(test_name="t", scenario="s", passed=False, turns_used=2, max_turns=2,
                            failure_reason="max turns exhausted", turns=TURNS, spans=(Span(),) * 40)
    assert "<SPAN>" not in repr(result) and "tester_message" not in repr(result)
    assert "max turns exhausted" in repr(result)


def test_the_judge_sees_the_conversation_ending_with_what_the_user_just_said():
    assert ScenarioHarness._transcript(TURNS[:1], 2, "Seattle").splitlines() == [
        "turn 1 user> I need a flight out of San Francisco.", "turn 1 agent> Where to?", "turn 2 user> Seattle"]
