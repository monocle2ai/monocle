"""The input/output assertions read each agent's own expectation from the test case.

Every entry in tc.agents is checked -- duplicates included -- against the single
span list its name resolved to, and every failure is reported in one
AssertionError, since record_assertion keeps only the first per chain.
"""
import os

import pytest

from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.span_loader import JSONSpanLoader

TRACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", "trace1.json")
SUPERVISOR = "adk_supervisor_agent_5"
HOTEL = "adk_hotel_booking_agent_5"
SUPERVISOR_OUT = "OK. Here's a summary of your travel arrangements:"
HOTEL_OUT = "OK. I have booked a stay at Marriot Intercontinental"
TURN_INPUT = "Book a flight from San Francisco to Mumbai"


@pytest.fixture(autouse=True)
def _reset():
    TraceAssertion._assertion_errors = []
    TraceAssertion._entity_spans = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._entity_spans = None


@pytest.fixture(name="asserter")
def asserter_fixture():
    a = TraceAssertion.get_trace_asserter()
    a.load_spans(JSONSpanLoader.from_json(TRACE))
    yield a
    a.cleanup()


def _drain():
    messages = [e["message"] for e in TraceAssertion._assertion_errors]
    TraceAssertion._assertion_errors = []
    return messages


def _select(asserter, testcase):
    """called_agent first -- the I/O checks read the map it builds."""
    scope = asserter.called_agent(testcase=testcase)
    _drain()
    return scope


def _recorded(scope, name, event_name, attr):
    """The value one of an agent's spans actually recorded."""
    from monocle_test_tools.testcase import Agent

    span = scope._entity_span_list(Agent(name=name))[0]
    return [e.attributes.get(attr) for e in span.events if e.name == event_name][0]


class TestContainsOutput:

    def test_passes_when_each_agent_output_matches(self, asserter):
        tc = {"agents": {SUPERVISOR: {"output": SUPERVISOR_OUT},
                         HOTEL: {"output": HOTEL_OUT}}}

        _select(asserter, tc).contains_output(testcase=tc)

        assert _drain() == []

    def test_fails_naming_the_agent(self, asserter):
        tc = {"agents": {SUPERVISOR: {"output": "something else entirely"}}}

        _select(asserter, tc).contains_output(testcase=tc)

        messages = _drain()
        assert len(messages) == 1
        assert SUPERVISOR in messages[0]

    def test_every_failing_agent_is_reported_in_one_failure(self, asserter):
        tc = {"agents": {SUPERVISOR: {"output": "wrong one"},
                         HOTEL: {"output": "wrong two"}}}

        _select(asserter, tc).contains_output(testcase=tc)

        messages = _drain()
        assert len(messages) == 1
        assert SUPERVISOR in messages[0] and HOTEL in messages[0]

    def test_agents_without_an_output_are_skipped(self, asserter):
        tc = {"agents": {SUPERVISOR: {"output": SUPERVISOR_OUT}, HOTEL: {}}}

        _select(asserter, tc).contains_output(testcase=tc)

        assert _drain() == []

    def test_all_outputs_unset_is_ignored(self, asserter):
        """A test case that states no output expectation fails no output check."""
        tc = {"agents": {SUPERVISOR: {}, HOTEL: {}}}

        _select(asserter, tc).contains_output(testcase=tc)

        assert _drain() == []

    def test_an_empty_output_is_ignored(self, asserter):
        """"" states no expectation either, so it must not be checked as a value."""
        tc = {"agents": {SUPERVISOR: {"output": ""}}}

        _select(asserter, tc).contains_output(testcase=tc)

        assert _drain() == []

    def test_an_unset_output_does_not_mask_a_set_one(self, asserter):
        """Skipping the silent entries must not skip the real check."""
        tc = {"agents": {SUPERVISOR: {"output": "definitely not present"}, HOTEL: {}}}

        _select(asserter, tc).contains_output(testcase=tc)

        messages = _drain()
        assert len(messages) == 1
        assert SUPERVISOR in messages[0]

    def test_an_agent_missing_from_the_map_is_skipped(self, asserter):
        """It already failed in called_agent; do not report it again here."""
        tc = {"agents": {SUPERVISOR: {"output": SUPERVISOR_OUT},
                         "nope": {"output": "anything"}}}

        _select(asserter, tc).contains_output(testcase=tc)

        assert _drain() == []

    def test_without_called_agent_first_raises(self, asserter):
        tc = {"agents": {SUPERVISOR: {"output": SUPERVISOR_OUT}}}

        with pytest.raises(ValueError, match="called_agent"):
            asserter.contains_output(testcase=tc)


class TestDuplicateAgentEntries:
    """Both entries for one name are checked against that name's single span list."""

    def test_both_matching_entries_pass(self, asserter):
        tc = {"agents": [{SUPERVISOR: {"output": "OK. Here's a summary"}},
                         {SUPERVISOR: {"output": "travel arrangements"}}]}

        _select(asserter, tc).contains_output(testcase=tc)

        assert _drain() == []

    def test_one_bad_entry_of_a_pair_fails_once(self, asserter):
        tc = {"agents": [{SUPERVISOR: {"output": "OK. Here's a summary"}},
                         {SUPERVISOR: {"output": "definitely not present"}}]}

        _select(asserter, tc).contains_output(testcase=tc)

        messages = _drain()
        assert len(messages) == 1
        assert "definitely not present" in messages[0]


class TestTheOtherSevenMethods:

    def test_contains_input(self, asserter):
        tc = {"agents": {SUPERVISOR: {"input": TURN_INPUT}}}

        _select(asserter, tc).contains_input(testcase=tc)

        assert _drain() == []

    def test_has_output(self, asserter):
        """has_* uses the configured comparer, so the value must match in full."""
        scope = _select(asserter, {"agents": {HOTEL: {}}})
        recorded = _recorded(scope, HOTEL, "data.output", "response")
        tc = {"agents": {HOTEL: {"output": recorded}}}

        _select(asserter, tc).has_output(testcase=tc)

        assert _drain() == []

    def test_has_input(self, asserter):
        scope = _select(asserter, {"agents": {HOTEL: {}}})
        recorded = _recorded(scope, HOTEL, "data.input", "input")
        tc = {"agents": {HOTEL: {"input": recorded}}}

        _select(asserter, tc).has_input(testcase=tc)

        assert _drain() == []

    def test_does_not_contain_output_passes_when_absent(self, asserter):
        tc = {"agents": {SUPERVISOR: {"output": "text that is definitely absent"}}}

        _select(asserter, tc).does_not_contain_output(testcase=tc)

        assert _drain() == []

    def test_does_not_contain_output_fails_when_present(self, asserter):
        tc = {"agents": {SUPERVISOR: {"output": SUPERVISOR_OUT}}}

        _select(asserter, tc).does_not_contain_output(testcase=tc)

        assert len(_drain()) == 1

    def test_does_not_contain_input_passes_when_absent(self, asserter):
        tc = {"agents": {SUPERVISOR: {"input": "text that is definitely absent"}}}

        _select(asserter, tc).does_not_contain_input(testcase=tc)

        assert _drain() == []

    def test_does_not_have_output_passes_when_different(self, asserter):
        tc = {"agents": {SUPERVISOR: {"output": "not the recorded output"}}}

        _select(asserter, tc).does_not_have_output(testcase=tc)

        assert _drain() == []

    def test_does_not_have_input_passes_when_different(self, asserter):
        tc = {"agents": {SUPERVISOR: {"input": "not the recorded input"}}}

        _select(asserter, tc).does_not_have_input(testcase=tc)

        assert _drain() == []


class TestRejectedArguments:

    def test_value_and_testcase_together_raise(self, asserter):
        tc = {"agents": {SUPERVISOR: {"output": SUPERVISOR_OUT}}}
        scope = _select(asserter, tc)

        with pytest.raises(ValueError, match="cannot be combined with 'testcase'"):
            scope.contains_output("something", testcase=tc)

    def test_no_value_and_no_testcase_raises(self, asserter):
        with pytest.raises(ValueError, match="expected_output_substring"):
            asserter.contains_output()

    @pytest.mark.parametrize("method", [
        "has_any_input", "has_any_output", "contains_any_input", "contains_any_output",
        "does_not_have_any_input", "does_not_have_any_output",
        "does_not_contain_any_input", "does_not_contain_any_output",
    ])
    def test_any_variants_reject_testcase(self, asserter, method):
        tc = {"agents": {SUPERVISOR: {"output": SUPERVISOR_OUT}}}

        with pytest.raises(ValueError, match="does not support 'testcase'"):
            getattr(asserter, method)(testcase=tc)


def test_success_criterion(asserter):
    """The spec's success criterion, verbatim."""
    tc = {"agents": {SUPERVISOR: {"output": SUPERVISOR_OUT},
                     HOTEL: {"output": HOTEL_OUT}}}

    asserter.called_agent(testcase=tc).contains_output(testcase=tc)

    assert _drain() == []


class TestTopLevelOutput:
    """A test case may state an expected output without naming an entity.

    That is a plain end-to-end check: no called_agent/called_tool in front, so
    the value is checked against whatever spans are in scope.
    """

    def test_a_string_is_checked_against_the_trace(self, asserter):
        tc = {"output": SUPERVISOR_OUT}

        asserter.contains_output(testcase=tc)

        assert _drain() == []

    def test_every_entry_of_a_list_must_appear(self, asserter):
        tc = {"output": ["OK. Here's a summary", "Marriot Intercontinental"]}

        asserter.contains_output(testcase=tc)

        assert _drain() == []

    def test_a_missing_entry_fails(self, asserter):
        tc = {"output": ["OK. Here's a summary", "definitely not present"]}

        asserter.contains_output(testcase=tc)

        messages = _drain()
        assert len(messages) == 1
        assert "definitely not present" in messages[0]

    def test_it_lives_under_the_expected_wrapper_too(self, asserter):
        tc = {"expected": {"output": SUPERVISOR_OUT}}

        asserter.contains_output(testcase=tc)

        assert _drain() == []

    def test_entities_win_when_a_selector_ran(self, asserter):
        """With a selector in front, the per-entity expectations are what count."""
        tc = {"agents": {SUPERVISOR: {"output": SUPERVISOR_OUT}},
              "output": "definitely not present"}

        _select(asserter, tc).contains_output(testcase=tc)

        assert _drain() == []

    def test_no_output_and_no_selector_still_raises(self, asserter):
        with pytest.raises(ValueError, match="called_agent"):
            asserter.contains_output(testcase={"agents": {SUPERVISOR: {}}})

    def test_has_output_uses_the_configured_comparer(self, asserter):
        tc = {"output": "not the recorded output in full"}

        asserter.has_output(testcase=tc)

        assert len(_drain()) == 1

    def test_does_not_contain_output_inverts_it(self, asserter):
        tc = {"output": "text that is definitely absent"}

        asserter.does_not_contain_output(testcase=tc)

        assert _drain() == []
