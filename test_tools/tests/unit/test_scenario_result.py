"""Tests for the scenario result and its failure report. Pure formatting."""
from monocle_test_tools.harness.evaluator import Verdict
from monocle_test_tools.harness.param_tools import ParamCall
from monocle_test_tools.harness.result import ScenarioResult, TurnRecord


def _failed_result():
    return ScenarioResult(
        test_name="flight",
        scenario="Book a flight from source to destination.",
        passed=False,
        turns_used=2,
        max_turns=2,
        failure_reason="max turns exhausted",
        turns=[
            TurnRecord(turn=1, tester_message="I need a flight out of San Francisco.",
                       target_response="Where would you like to fly to?",
                       verdict=Verdict(met=False, reason="no destination captured yet")),
            TurnRecord(turn=2, tester_message="Seattle, obviously.",
                       target_response="What date?",
                       verdict=Verdict(met=False, reason="no booking confirmation"),
                       param_tools_called=[
                           ParamCall(name="destination", value="Seattle")]),
        ],
    )


def test_report_leads_with_the_outcome_and_turn_count():
    report = _failed_result().report()
    assert report.startswith(
        "FAILED scenario 'Book a flight from source to destination.' after 2/2 turns")


def test_report_states_the_failure_reason():
    assert "reason: max turns exhausted" in _failed_result().report()


def test_report_states_the_last_verdict():
    assert "last verdict: no booking confirmation" in _failed_result().report()


def test_report_includes_every_turn():
    report = _failed_result().report()
    assert "turn 1" in report
    assert "turn 2" in report
    assert "I need a flight out of San Francisco." in report
    assert "Where would you like to fly to?" in report


def test_report_puts_each_tool_call_on_its_own_line_with_its_value():
    lines = _failed_result().report().splitlines()
    assert "  turn 2  tester> [called destination()] ==> Seattle" in lines


def test_tool_call_line_precedes_the_tester_message_which_is_unprefixed():
    lines = _failed_result().report().splitlines()
    call_line = lines.index("  turn 2  tester> [called destination()] ==> Seattle")
    assert lines[call_line + 1] == "          tester> Seattle, obviously."


def test_turn_number_appears_once_per_turn_on_the_first_line():
    lines = [line for line in _failed_result().report().splitlines()
             if "turn 2" in line]
    assert len(lines) == 1


def test_a_turn_with_no_tool_calls_puts_the_number_on_the_message_line():
    lines = _failed_result().report().splitlines()
    assert "  turn 1  tester> I need a flight out of San Francisco." in lines


def test_every_tool_call_in_a_turn_gets_its_own_line():
    result = ScenarioResult(
        test_name="flight", scenario="Book a flight.", passed=True,
        turns_used=1, max_turns=8,
        turns=[TurnRecord(
            turn=1, tester_message="From San Francisco to Seattle.",
            target_response="Booked.",
            verdict=Verdict(met=True, reason="done"),
            param_tools_called=[ParamCall(name="source", value="San Francisco"),
                                ParamCall(name="destination", value="Seattle")])])
    lines = result.report().splitlines()
    assert "  turn 1  tester> [called source()] ==> San Francisco" in lines
    assert "          tester> [called destination()] ==> Seattle" in lines
    assert "          tester> From San Francisco to Seattle." in lines


def test_report_of_a_passing_result_says_passed():
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=True,
                            turns_used=1, max_turns=8,
                            turns=[TurnRecord(turn=1, tester_message="hi",
                                              target_response="booked",
                                              verdict=Verdict(met=True, reason="done"))])
    assert result.report().startswith("PASSED scenario 'Book a flight.' after 1/8 turns")


def test_report_handles_a_turn_with_no_verdict():
    """A turn that ended in a target error records no verdict."""
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=False,
                            turns_used=1, max_turns=8,
                            failure_reason="target error: boom",
                            turns=[TurnRecord(turn=1, tester_message="hi")])
    report = result.report()
    assert "target error: boom" in report
    assert "turn 1" in report


def test_report_of_a_result_with_no_turns_is_still_readable():
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=False,
                            turns_used=0, max_turns=8, failure_reason="target error: boom")
    assert "FAILED" in result.report()


# --- missing required params ----------------------------------------------

def test_report_names_required_params_that_were_never_requested():
    result = ScenarioResult(
        test_name="flight", scenario="Book a flight.", passed=False,
        turns_used=1, max_turns=8,
        failure_reason="required params never requested: date",
        missing_required_params=["date"],
        turns=[TurnRecord(turn=1, tester_message="hi", target_response="booked",
                          verdict=Verdict(met=True, reason="confirmed"))])
    lines = result.report().splitlines()
    # A dedicated line, not just the substring the failure_reason line already carries.
    assert "  never requested: date" in lines
    assert lines[0].startswith("FAILED")


def test_report_omits_the_line_when_nothing_is_missing():
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=True,
                            turns_used=1, max_turns=8,
                            turns=[TurnRecord(turn=1, tester_message="hi",
                                              target_response="booked",
                                              verdict=Verdict(met=True, reason="done"))])
    assert "never requested" not in result.report()


def test_missing_required_params_defaults_to_empty():
    result = ScenarioResult(test_name="flight", scenario="Book a flight.", passed=True,
                            turns_used=1, max_turns=8)
    assert result.missing_required_params == []


# --- repr stays readable in pytest output ----------------------------------

class _FakeSpan:
    """Stands in for a ReadableSpan, whose repr is what floods the terminal."""

    def __repr__(self):
        return "<SPAN-REPR-MARKER>"


def _result_with_bulky_fields():
    spans = tuple(_FakeSpan() for _ in range(40))
    return ScenarioResult(
        test_name="book_flight_foul_mouthed_user",
        scenario="Book a flight from source to destination.",
        passed=False, turns_used=5, max_turns=8,
        failure_reason="required params never requested: date",
        missing_required_params=["date"],
        turns=[TurnRecord(turn=1,
                          tester_message="I need a damn flight out of San Francisco.",
                          target_response="Where would you like to fly to?",
                          verdict=Verdict(met=False, reason="not booked"))],
        spans=spans, per_turn_spans=[spans])


def test_repr_does_not_dump_spans():
    """pytest prints repr(result) on `assert result.passed` -- spans must stay out."""
    assert "SPAN-REPR-MARKER" not in repr(_result_with_bulky_fields())


def test_repr_does_not_dump_the_turn_transcript():
    """The transcript belongs in report(), which is the assert message."""
    assert "tester_message" not in repr(_result_with_bulky_fields())


def test_repr_keeps_the_fields_that_identify_the_failure():
    text = repr(_result_with_bulky_fields())
    assert "passed=False" in text
    assert "required params never requested: date" in text
    assert "book_flight_foul_mouthed_user" in text


def test_repr_is_short_enough_to_read_in_a_terminal():
    assert len(repr(_result_with_bulky_fields())) < 400


def test_report_still_shows_the_full_transcript():
    """Hiding fields from repr must not touch report()."""
    assert "I need a damn flight" in _result_with_bulky_fields().report()
