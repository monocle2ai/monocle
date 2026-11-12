"""A failed scenario explains itself: the reason and every turn, with no span noise in the assert message."""
import os

from monocle_test_tools import scenario_results
from monocle_test_tools.harness.driver import ScenarioHarness
from monocle_test_tools.harness.evaluator import Verdict
from monocle_test_tools.harness.result import ScenarioResult, TurnRecord
from monocle_test_tools.harness.test_agent import ParamCall

TURNS = [TurnRecord(turn=1, tester_message="I need a flight out of San Francisco.", target_response="Where to?",
                    verdict=Verdict(met=False, reason="no destination yet")),
         TurnRecord(turn=2, tester_message="Seattle, obviously.", target_response="What date?",
                    verdict=Verdict(met=False, reason="not booked"),
                    param_tools_called=[ParamCall(name="destination", value="Seattle")])]


def failed(**overrides) -> ScenarioResult:
    fields = dict(test_name="book_flight.withhold.date", scenario="Book a flight.", passed=False, turns_used=2,
                  max_turns=2, failure_reason="criteria not met in 2 turns", reason="not booked",
                  success_criteria="Ask for the date.", session_id="s-1", turns=TURNS)
    return ScenarioResult(**{**fields, **overrides})


def test_the_report_says_what_went_wrong_then_where_to_look_then_each_turn():
    lines = failed().report().splitlines()
    assert lines[:5] == ["criteria not met in 2 turns",
                         "  reason    not booked",
                         "  criteria  Ask for the date.",
                         "  session   s-1", ""]
    assert lines[lines.index("  turn 2") + 1:] == [
        "    param  destination = Seattle", "    user   Seattle, obviously.",
        "    agent  What date?", "    judge  [not met] not booked"]


def test_the_report_shows_each_turns_trace_and_lines_up_multiline_replies():
    turn = TurnRecord(turn=1, tester_message="Hi", target_response="Sure.\nWhich date?", trace_id="ab" * 16,
                      verdict=Verdict(met=False, violated=True, reason="booked anyway"))
    lines = failed(turns=[turn]).report(saved="x.log").splitlines()
    assert "  saved     x.log" in lines
    assert f"  turn 1  trace {'ab' * 16}" in lines
    assert lines[-3:] == ["    agent  Sure.", "           Which date?", "    judge  [violated] booked anyway"]


def test_an_unjudged_conversation_shows_each_reply_and_only_the_failed_turn_as_an_error():
    turns = [TurnRecord(turn=1, tester_message="Hi", target_response="Hello."),
             TurnRecord(turn=2, tester_message="Book it.")]
    lines = failed(turns=turns, success_criteria=None, failure_reason="agent raised an error at turn 2 of 8",
                   reason="boom").report().splitlines()
    assert lines[lines.index("  turn 1") + 1:lines.index("  turn 2")] == ["    user   Hi", "    agent  Hello.", ""]
    assert lines[-1] == "    agent  (error, see reason)"
    assert not any("judge" in line for line in lines)


def test_results_are_logged_only_when_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(scenario_results.RESULTS_ENV, raising=False)
    scenario_results.reset()
    assert scenario_results.record(failed()) is None

    monkeypatch.setenv(scenario_results.RESULTS_ENV, "true")
    try:
        path = scenario_results.record(failed(test_name="ok", passed=True, turns_used=1))
        assert scenario_results.record(failed()) == path
    finally:
        scenario_results.reset()
    assert os.path.dirname(path) == os.path.join(".", ".monocle", "test_results")
    assert open(path).read().splitlines()[:4] == ["PASSED  ok",
                                                  "FAILED  book_flight.withhold.date",
                                                  "  criteria not met in 2 turns",
                                                  "    reason    not booked"]


def test_repr_leaves_out_spans_and_turns():
    class Span:
        def __repr__(self):
            return "<SPAN>"
    result = failed(spans=(Span(),) * 40)
    assert "<SPAN>" not in repr(result) and "tester_message" not in repr(result)
    assert "criteria not met in 2 turns" in repr(result)


def test_the_judge_sees_the_conversation_ending_with_what_the_user_just_said():
    assert ScenarioHarness._transcript(TURNS[:1], 2, "Seattle").splitlines() == [
        "turn 1 user> I need a flight out of San Francisco.", "turn 1 agent> Where to?", "turn 2 user> Seattle"]
