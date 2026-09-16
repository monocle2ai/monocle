"""called_agent(testcase=) resolves every agent the test case names.

One span list per DISTINCT agent name: FluentTestCase.from_spans deliberately
keeps same-name agents with different input/output as separate entries, so a
discovered test case routinely carries duplicates that all describe one agent.
"""
import os

import pytest

from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.span_loader import JSONSpanLoader

TRACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", "trace1.json")
SUPERVISOR = "adk_supervisor_agent_5"
HOTEL = "adk_hotel_booking_agent_5"
FLIGHT = "adk_flight_booking_agent_5"


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


def _names(scope):
    return [entity.name for entity, _ in scope._entity_spans]


class TestMapConstruction:

    def test_every_named_agent_gets_an_entry(self, asserter):
        scope = asserter.called_agent(testcase={"agents": {SUPERVISOR: {}, HOTEL: {}}})

        assert _drain() == []
        assert _names(scope) == [SUPERVISOR, HOTEL]

    def test_entries_hold_that_agents_spans(self, asserter):
        scope = asserter.called_agent(testcase={"agents": {SUPERVISOR: {}}})

        spans = dict((e.name, s) for e, s in scope._entity_spans)[SUPERVISOR]
        assert spans
        assert all(s.attributes.get("entity.1.name") == SUPERVISOR for s in spans)

    def test_filtered_spans_is_the_union(self, asserter):
        scope = asserter.called_agent(testcase={"agents": {SUPERVISOR: {}, HOTEL: {}}})

        total = sum(len(s) for _, s in scope._entity_spans)
        assert len(scope._filtered_spans) == total

    def test_a_missing_agent_fails_and_is_excluded(self, asserter):
        scope = asserter.called_agent(testcase={"agents": {SUPERVISOR: {}, "nope": {}}})

        assert any("nope" in m for m in _drain())
        assert _names(scope) == [SUPERVISOR]

    def test_several_missing_agents_are_reported_in_one_failure(self, asserter):
        asserter.called_agent(testcase={"agents": {"nope1": {}, "nope2": {}}})

        messages = _drain()
        assert len(messages) == 1
        assert "nope1" in messages[0] and "nope2" in messages[0]


class TestDuplicateAgentNames:
    """Two entries for one name share a single span list."""

    DUPES = {"agents": [
        {SUPERVISOR: {"output": "OK. Here's a summary"}},
        {SUPERVISOR: {"output": "travel arrangements"}},
    ]}

    def test_one_map_entry_per_distinct_name(self, asserter):
        scope = asserter.called_agent(testcase=self.DUPES)

        assert _drain() == []
        assert _names(scope) == [SUPERVISOR]

    def test_a_missing_duplicated_name_is_reported_once(self, asserter):
        asserter.called_agent(testcase={"agents": [{"nope": {"output": "a"}},
                                                   {"nope": {"output": "b"}}]})

        messages = _drain()
        assert len(messages) == 1
        assert messages[0].count("nope") == 1

    def test_union_does_not_double_count_a_duplicated_name(self, asserter):
        once = asserter.called_agent(testcase={"agents": {SUPERVISOR: {}}})
        _drain()
        twice = asserter.called_agent(testcase=self.DUPES)
        _drain()

        assert len(twice._filtered_spans) == len(once._filtered_spans)


class TestRejectedArguments:

    @pytest.mark.parametrize("conflicting", [
        {"agent_name": "x"}, {"count": 1}, {"min_count": 1}, {"max_count": 2},
    ])
    def test_single_agent_arguments_are_rejected(self, asserter, conflicting):
        with pytest.raises(ValueError, match="cannot be combined with 'testcase'"):
            asserter.called_agent(testcase={"agents": {SUPERVISOR: {}}}, **conflicting)

    def test_empty_agents_raises(self, asserter):
        with pytest.raises(ValueError, match="no agents"):
            asserter.called_agent(testcase={"evals": {"hallucination": "minor"}})


def test_without_testcase_behaves_as_before(asserter):
    asserter.called_agent(FLIGHT)

    assert _drain() == []
