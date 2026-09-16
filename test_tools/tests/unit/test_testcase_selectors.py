"""Chain rules for testcase-driven selectors.

Each testcase-driven selector builds its own entity->spans map, so two different
selectors in one chain is ambiguous. The rule is scoped to testcase mode: the
plain called_agent(A).called_tool(T) narrowing that CLAUDE.md documents and
test_performance.py exercises must keep working.
"""
import os

import pytest

from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.span_loader import JSONSpanLoader

TRACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", "trace1.json")
AGENT = "adk_supervisor_agent_5"
# The one tool in trace1.json, and the agent that actually calls it -- the plain
# narrowing chain looks for the tool *inside* the agent, so they must match.
TOOL = "adk_book_hotel_5"
TOOL_AGENT = "adk_hotel_booking_agent_5"
TESTCASE = {"agents": {AGENT: {}}}


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


class TestSelectorRule:

    def test_agent_then_tool_raises(self, asserter):
        scope = asserter.called_agent(testcase=TESTCASE)
        _drain()

        with pytest.raises(ValueError, match="selector"):
            scope.called_tool(testcase=TESTCASE)

    def test_plain_agent_then_tool_is_not_blocked(self, asserter):
        """The rule must not fire for a chain that passes no testcase.

        Asserts only that no ValueError escapes -- not that the chain finds
        anything. Narrowing is a flat filter over the span list
        (validator._filter_spans_by_type), so called_agent leaves only
        agentic.invocation spans and a following called_tool can never match a
        agentic.tool.invocation one. That is a pre-existing gap between
        CLAUDE.md's "tool T inside agent A" and the implementation; the working
        form is called_tool(T, agent_name=A). Not this rule's business either
        way -- what matters is that the rule stays out of plain chains.
        """
        asserter.called_agent(TOOL_AGENT).called_tool(TOOL)

        _drain()

    def test_plain_tool_scoped_by_agent_still_works(self, asserter):
        """The form that does work is untouched."""
        asserter.called_tool(TOOL, agent_name=TOOL_AGENT)

        assert _drain() == []

    def test_same_selector_twice_is_allowed(self, asserter):
        """Two called_agent calls are one selector kind, so the rule does not fire."""
        scope = asserter.called_agent(testcase=TESTCASE)
        scope.called_agent(testcase=TESTCASE)

        _drain()  # presence assertions may or may not fail; the rule must not raise


class TestCalledToolTestcase:

    def test_a_testcase_with_no_tools_raises(self, asserter):
        """TESTCASE names only agents, so a tool selector has nothing to select."""
        with pytest.raises(ValueError, match="no tools"):
            asserter.called_tool(testcase=TESTCASE)

    def test_a_tool_testcase_selects(self, asserter):
        scope = asserter.called_tool(testcase={"tools": {TOOL: {}}})

        assert _drain() == []
        assert [e.name for e, _ in scope._entity_spans] == [TOOL]


class TestStateThreading:

    def test_entity_spans_defaults_to_none(self):
        assert TraceAssertion()._entity_spans is None

    def test_entity_spans_is_carried_down_a_chain(self, asserter):
        # An output is needed: a check where no agent sets the field raises
        # rather than passing vacuously, which would mask the threading.
        testcase = {"agents": {AGENT: {"output": "OK. Here's a summary"}}}
        scope = asserter.called_agent(testcase=testcase)
        _drain()

        assert scope._entity_spans is not None
        assert scope.contains_output(testcase=testcase)._entity_spans is scope._entity_spans

    def test_cleanup_clears_entity_spans(self):
        TraceAssertion._entity_spans = [("stale", [])]

        TraceAssertion().cleanup()

        assert TraceAssertion._entity_spans is None
