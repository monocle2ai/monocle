"""The scenario result's failure report and its terminal-safe repr. Pure formatting."""
from monocle_test_tools.harness.evaluator import Verdict
from monocle_test_tools.harness.param_tools import ParamCall
from monocle_test_tools.harness.result import ScenarioResult, TurnRecord


def _failed_result():
    return ScenarioResult(
        test_name="flight", scenario="Book a flight from source to destination.",
        passed=False, turns_used=2, max_turns=2, failure_reason="max turns exhausted",
        turns=[
            TurnRecord(turn=1, tester_message="I need a flight out of San Francisco.",
                       target_response="Where would you like to fly to?",
                       verdict=Verdict(met=False, reason="no destination captured yet")),
            TurnRecord(turn=2, tester_message="Seattle, obviously.",
                       target_response="What date?",
                       verdict=Verdict(met=False, reason="no booking confirmation"),
                       param_tools_called=[ParamCall(name="destination", value="Seattle")]),
        ])


def test_report_leads_with_outcome_reason_and_last_verdict():
    report = _failed_result().report()
    assert report.startswith("FAILED scenario 'Book a flight from source to destination.' after 2/2 turns")
    assert "reason: max turns exhausted" in report
    assert "last verdict: no booking confirmation" in report


def test_report_shows_every_turn_with_tool_calls_on_their_own_lines():
    lines = _failed_result().report().splitlines()
    assert "  turn 1  tester> I need a flight out of San Francisco." in lines
    call_line = lines.index("  turn 2  tester> [called destination()] ==> Seattle")
    assert lines[call_line + 1] == "          tester> Seattle, obviously."
    assert sum("turn 2" in line for line in lines) == 1


def test_report_of_a_passing_result_says_passed():
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=True,
                            turns_used=1, max_turns=8,
                            turns=[TurnRecord(turn=1, tester_message="hi", target_response="booked",
                                              verdict=Verdict(met=True, reason="done"))])
    assert result.report().startswith("PASSED scenario 'Book a flight.' after 1/8 turns")
    assert "never requested" not in result.report()


def test_report_handles_a_target_error_with_no_verdict():
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=False,
                            turns_used=1, max_turns=8, failure_reason="target error: boom",
                            turns=[TurnRecord(turn=1, tester_message="hi")])
    assert "target error: boom" in result.report() and "turn 1" in result.report()


def test_report_names_required_params_that_were_never_requested():
    result = ScenarioResult(
        test_name="flight", scenario="Book a flight.", passed=False, turns_used=1, max_turns=8,
        failure_reason="required params never requested: date", missing_required_params=["date"],
        turns=[TurnRecord(turn=1, tester_message="hi", target_response="booked",
                          verdict=Verdict(met=True, reason="confirmed"))])
    assert "  never requested: date" in result.report().splitlines()


class _FakeSpan:
    def __repr__(self):
        return "<SPAN-REPR-MARKER>"


def test_repr_hides_spans_and_transcript_but_keeps_the_failure():
    """pytest prints repr(result) on `assert result.passed`; it must stay readable."""
    spans = tuple(_FakeSpan() for _ in range(40))
    result = ScenarioResult(
        test_name="book_flight", scenario="Book a flight.", passed=False, turns_used=5, max_turns=8,
        failure_reason="required params never requested: date", missing_required_params=["date"],
        turns=[TurnRecord(turn=1, tester_message="I need a damn flight.", target_response="Where to?",
                          verdict=Verdict(met=False, reason="not booked"))],
        spans=spans, per_turn_spans=[spans])
    text = repr(result)
    assert "SPAN-REPR-MARKER" not in text and "tester_message" not in text
    assert "passed=False" in text and "never requested: date" in text and len(text) < 400
    assert "I need a damn flight." in result.report()


def test_the_judge_transcript_ends_with_what_the_user_just_said():
    from monocle_test_tools.harness.driver import ScenarioHarness
    records = [TurnRecord(turn=1, tester_message="book it", target_response="which date?",
                          verdict=Verdict(met=False, reason="asked"))]
    text = ScenarioHarness._transcript(records, 2, "22 Oct 2026, go ahead")
    assert text.splitlines() == ["turn 1 user> book it", "turn 1 agent> which date?",
                                 "turn 2 user> 22 Oct 2026, go ahead"]
