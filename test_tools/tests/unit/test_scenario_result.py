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
                  max_turns=2, failure_reason="max turns exhausted", success_criteria="Ask for the date.",
                  session_id="s-1", tester_model="gpt-4o-mini", judge_model="gpt-4o", turns=TURNS)
    return ScenarioResult(**{**fields, **overrides})


def test_the_report_leads_with_the_reason_then_where_to_look_then_each_turn():
    lines = failed().report().splitlines()
    assert lines[:8] == ["max turns exhausted", "",
                         "  test      book_flight.withhold.date",
                         "  turns     2 of 2",
                         "  criteria  Ask for the date.",
                         "  models    tester gpt-4o-mini, judge gpt-4o",
                         "  session   s-1", ""]
    assert lines[lines.index("  turn 2") + 1:lines.index("  turn 2") + 5] == [
        "    param  destination = Seattle", "    user   Seattle, obviously.",
        "    agent  What date?", "    judge  [not met] not booked"]


def test_the_report_shows_each_turns_trace_and_lines_up_multiline_replies():
    turn = TurnRecord(turn=1, tester_message="Hi", target_response="Sure.\nWhich date?", trace_id="ab" * 16,
                      verdict=Verdict(met=False, violated=True, reason="booked anyway"))
    lines = failed(turns=[turn]).report(saved="x/01_t.txt").splitlines()
    assert "  saved     x/01_t.txt" in lines
    assert f"  turn 1  trace {'ab' * 16}" in lines
    assert lines[-3:] == ["    agent  Sure.", "           Which date?", "    judge  [violated] booked anyway"]


def test_results_are_saved_only_when_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(scenario_results.RESULTS_ENV, raising=False)
    monkeypatch.setattr(scenario_results, "_enabled", False)
    monkeypatch.setattr(scenario_results, "_run_dir", None)
    monkeypatch.setattr(scenario_results, "_results", [])
    assert scenario_results.record(failed()) is None and scenario_results.run_dir() is None

    monkeypatch.setenv(scenario_results.RESULTS_ENV, "true")
    assert scenario_results.record(failed(test_name="ok", passed=True, turns_used=1)) is None
    path = scenario_results.record(failed())
    assert os.path.basename(path) == "02_book_flight.withhold.date.txt"
    assert os.path.dirname(path).startswith(os.path.join(".", ".monocle", "test_results"))
    assert open(path).read().startswith("max turns exhausted\n")
    summary = open(os.path.join(scenario_results.run_dir(), "summary.txt")).read().splitlines()
    assert summary[1:] == ["1 passed, 1 failed", "",
                           "   #  result  turns  test                       reason",
                           "   1  passed  1/2    ok",
                           "   2  FAILED  2/2    book_flight.withhold.date  max turns exhausted"]


def test_repr_leaves_out_spans_and_turns():
    class Span:
        def __repr__(self):
            return "<SPAN>"
    result = failed(spans=(Span(),) * 40)
    assert "<SPAN>" not in repr(result) and "tester_message" not in repr(result)
    assert "max turns exhausted" in repr(result)


def test_the_judge_sees_the_conversation_ending_with_what_the_user_just_said():
    assert ScenarioHarness._transcript(TURNS[:1], 2, "Seattle").splitlines() == [
        "turn 1 user> I need a flight out of San Francisco.", "turn 1 agent> Where to?", "turn 2 user> Seattle"]
