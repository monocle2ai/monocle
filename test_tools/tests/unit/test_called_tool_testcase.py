"""called_tool(testcase=) resolves every tool the test case names.

The agent counterpart keys one queue per distinct agent name. Tools key on the
tool AND its calling agent, because from_spans records that agent and the same
tool called by two agents is two different span sets -- unlike two same-name
agent entries, which describe one agent.
"""
import os

import pytest

from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.span_loader import JSONSpanLoader

TRACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", "trace2.json")
FLIGHT = "okahu-demo-lg-tool_book_flight"
FLIGHT_AGENT = "okahu-demo-lg-agent-air_travel_assistant"
HOTEL = "okahu-demo-lg-tool_book_hotel"
HOTEL_AGENT = "okahu-demo-lg-agent-lodging_assistant"
WEATHER = "demo_get_weather"
WEATHER_AGENT = "okahu-demo-lg-agent-weather_assistant"


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


def _keys(scope):
    return [(e.name, e.agent.name if getattr(e, "agent", None) else None)
            for e, _ in scope._entity_spans]


class TestMapConstruction:

    def test_every_named_tool_gets_an_entry(self, asserter):
        scope = asserter.called_tool(testcase={"tools": {FLIGHT: {}, HOTEL: {}}})

        assert _drain() == []
        assert _keys(scope) == [(FLIGHT, None), (HOTEL, None)]

    def test_entries_hold_that_tools_spans(self, asserter):
        scope = asserter.called_tool(testcase={"tools": {FLIGHT: {}}})

        spans = scope._entity_spans[0][1]
        assert spans
        assert all(s.attributes.get("entity.1.name") == FLIGHT for s in spans)

    def test_filtered_spans_is_the_union(self, asserter):
        scope = asserter.called_tool(testcase={"tools": {FLIGHT: {}, HOTEL: {}}})

        total = sum(len(s) for _, s in scope._entity_spans)
        assert len(scope._filtered_spans) == total

    def test_a_missing_tool_fails_and_is_excluded(self, asserter):
        scope = asserter.called_tool(testcase={"tools": {FLIGHT: {}, "nope": {}}})

        assert any("nope" in m for m in _drain())
        assert _keys(scope) == [(FLIGHT, None)]

    def test_several_missing_tools_are_reported_in_one_failure(self, asserter):
        asserter.called_tool(testcase={"tools": {"nope1": {}, "nope2": {}}})

        messages = _drain()
        assert len(messages) == 1
        assert "nope1" in messages[0] and "nope2" in messages[0]


class TestCallingAgentScoping:
    """The agent is part of the key, not decoration."""

    def test_the_agent_narrows_the_lookup(self, asserter):
        scope = asserter.called_tool(
            testcase={"tools": [{FLIGHT: {"agent": {FLIGHT_AGENT: {}}}}]})

        assert _drain() == []
        assert _keys(scope) == [(FLIGHT, FLIGHT_AGENT)]

    def test_a_wrong_agent_is_a_miss(self, asserter):
        asserter.called_tool(
            testcase={"tools": [{FLIGHT: {"agent": {HOTEL_AGENT: {}}}}]})

        messages = _drain()
        assert len(messages) == 1
        assert FLIGHT in messages[0] and HOTEL_AGENT in messages[0]

    def test_same_tool_under_two_agents_is_two_queues(self, asserter):
        """Unlike agents, two same-name tool entries can be different span sets."""
        scope = asserter.called_tool(testcase={"tools": [
            {HOTEL: {"agent": {HOTEL_AGENT: {}}}},
            {HOTEL: {"agent": {WEATHER_AGENT: {}}}},
        ]})

        _drain()  # the second entry is a miss in this trace
        assert (HOTEL, HOTEL_AGENT) in _keys(scope)

    def test_duplicate_entries_for_one_key_share_a_queue(self, asserter):
        scope = asserter.called_tool(testcase={"tools": [
            {HOTEL: {"agent": {HOTEL_AGENT: {}}, "output": "Marriott Central Mumbai"}},
            {HOTEL: {"agent": {HOTEL_AGENT: {}}, "output": "Weather Update"}},
        ]})

        assert _drain() == []
        assert _keys(scope) == [(HOTEL, HOTEL_AGENT)]

    def test_an_agentless_entry_matches_any_caller(self, asserter):
        scope = asserter.called_tool(testcase={"tools": {HOTEL: {}}})

        assert _drain() == []
        assert len(scope._entity_spans[0][1]) == 2


class TestRejectedArguments:

    @pytest.mark.parametrize("conflicting", [
        {"tool_name": "x"}, {"agent_name": "y"}, {"count": 1},
        {"min_count": 1}, {"max_count": 2},
    ])
    def test_single_tool_arguments_are_rejected(self, asserter, conflicting):
        with pytest.raises(ValueError, match="cannot be combined with 'testcase'"):
            asserter.called_tool(testcase={"tools": {FLIGHT: {}}}, **conflicting)

    def test_empty_tools_raises(self, asserter):
        with pytest.raises(ValueError, match="no tools"):
            asserter.called_tool(testcase={"agents": {"a": {}}})


class TestIOChecksReadTools:
    """After a tool selector, the I/O checks read tc.tools rather than tc.agents."""

    def test_contains_output_checks_each_tool(self, asserter):
        tc = {"tools": {FLIGHT: {"output": "Successfully booked a flight"},
                        WEATHER: {"output": "temperature"}}}

        asserter.called_tool(testcase=tc).contains_output(testcase=tc)

        assert _drain() == []

    def test_a_failing_tool_is_named(self, asserter):
        tc = {"tools": {FLIGHT: {"output": "not what it returned"}}}

        asserter.called_tool(testcase=tc).contains_output(testcase=tc)

        messages = _drain()
        assert len(messages) == 1
        assert FLIGHT in messages[0]

    def test_duplicate_entries_are_both_checked_against_one_queue(self, asserter):
        tc = {"tools": [
            {HOTEL: {"agent": {HOTEL_AGENT: {}}, "output": "Marriott Central Mumbai"}},
            {HOTEL: {"agent": {HOTEL_AGENT: {}}, "output": "Weather Update"}},
        ]}

        asserter.called_tool(testcase=tc).contains_output(testcase=tc)

        assert _drain() == []

    def test_contains_input_reads_the_tool_input(self, asserter):
        tc = {"tools": {WEATHER: {"input": "Mumbai"}}}

        asserter.called_tool(testcase=tc).contains_input(testcase=tc)

        assert _drain() == []

    def test_an_unset_output_is_still_ignored(self, asserter):
        tc = {"tools": {FLIGHT: {}}}

        asserter.called_tool(testcase=tc).contains_output(testcase=tc)

        assert _drain() == []


def test_without_testcase_behaves_as_before(asserter):
    asserter.called_tool(FLIGHT)

    assert _drain() == []
