"""called_any_agent / called_any_tool accept several names and hold if any matched.

Any-of counterparts of called_agent/called_tool: a run with more than one
acceptable route (route to the flight agent OR the train agent) must not be
asserted with an all-of selector. Counts constrain the TOTAL invocations across
the named entities, so the assertions read the same as the single-name ones.

The negatives, does_not_call_any_*, are the mirror: they fail if ANY name ran,
which is how a set of forbidden routes is stated, and their failures name the
entities that ran rather than the candidates that did not.
"""
import os

import pytest

from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.span_loader import JSONSpanLoader

TRACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", "trace2.json")
SUPERVISOR = "okahu-demo-lg-agent-travel_supervisor"      # 4 invocations
AIR = "okahu-demo-lg-agent-air_travel_assistant"          # 1 invocation
LODGING = "okahu-demo-lg-agent-lodging_assistant"         # 1 invocation
BOOK_FLIGHT = "okahu-demo-lg-tool_book_flight"            # 2 calls, by AIR
BOOK_HOTEL = "okahu-demo-lg-tool_book_hotel"              # 2 calls, by LODGING
WEATHER = "demo_get_weather"                              # 1 call


@pytest.fixture(autouse=True)
def _reset():
    TraceAssertion._assertion_errors = []
    yield
    TraceAssertion._assertion_errors = []


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
    return [span.attributes.get("entity.1.name") for span in scope._filtered_spans]


class TestCalledAnyAgent:

    def test_passes_when_one_of_the_names_was_called(self, asserter):
        asserter.called_any_agent(LODGING, "never_ran")

        assert _drain() == []

    def test_fails_naming_every_candidate_when_none_ran(self, asserter):
        asserter.called_any_agent("never_ran", "also_never_ran")

        messages = _drain()
        assert len(messages) == 1
        assert "never_ran" in messages[0] and "also_never_ran" in messages[0]

    def test_narrows_to_the_union_of_matched_spans(self, asserter):
        scope = asserter.called_any_agent(AIR, LODGING)

        assert _drain() == []
        assert sorted(_names(scope)) == [AIR, LODGING]

    def test_a_list_argument_is_accepted(self, asserter):
        scope = asserter.called_any_agent([AIR, LODGING])

        assert _drain() == []
        assert len(scope._filtered_spans) == 2

    def test_counts_are_totals_across_the_named_agents(self, asserter):
        # One invocation each: the total, not the per-agent count, is what holds.
        asserter.called_any_agent(AIR, LODGING, count=2)
        assert _drain() == []

        asserter.called_any_agent(AIR, LODGING, count=1)
        messages = _drain()
        assert len(messages) == 1
        assert "2 times in total" in messages[0] and "exactly 1" in messages[0]

    def test_min_and_max_counts(self, asserter):
        asserter.called_any_agent(SUPERVISOR, AIR, min_count=5, max_count=5)
        assert _drain() == []

        asserter.called_any_agent(AIR, LODGING, min_count=3)
        messages = _drain()
        assert len(messages) == 1
        assert "2 times in total" in messages[0] and "at least 3" in messages[0]

        asserter.called_any_agent(SUPERVISOR, max_count=3)
        messages = _drain()
        assert len(messages) == 1
        assert "4 times in total" in messages[0] and "at most 3" in messages[0]

    def test_repeated_names_do_not_double_count(self, asserter):
        asserter.called_any_agent(AIR, AIR, count=1)

        assert _drain() == []

    def test_custom_message_replaces_the_default(self, asserter):
        asserter.called_any_agent("never_ran", message="NO ROUTE TAKEN")

        assert _drain() == ["NO ROUTE TAKEN"]

    def test_chains_onto_an_output_check(self, asserter):
        asserter.called_any_agent(LODGING, "never_ran").contains_output("Marriott")

        assert _drain() == []


class TestCalledAnyTool:

    def test_passes_when_one_of_the_names_was_called(self, asserter):
        asserter.called_any_tool(BOOK_FLIGHT, "never_ran")

        assert _drain() == []

    def test_fails_naming_every_candidate_when_none_ran(self, asserter):
        asserter.called_any_tool("never_ran", "also_never_ran")

        messages = _drain()
        assert len(messages) == 1
        assert "never_ran" in messages[0] and "also_never_ran" in messages[0]

    def test_narrows_to_the_union_in_trace_order(self, asserter):
        scope = asserter.called_any_tool(BOOK_FLIGHT, WEATHER)

        assert _drain() == []
        assert _names(scope) == [BOOK_FLIGHT, BOOK_FLIGHT, WEATHER]

    def test_a_list_argument_is_accepted(self, asserter):
        scope = asserter.called_any_tool([BOOK_FLIGHT, WEATHER])

        assert _drain() == []
        assert len(scope._filtered_spans) == 3

    def test_agent_name_restricts_the_caller(self, asserter):
        scope = asserter.called_any_tool(BOOK_FLIGHT, BOOK_HOTEL, agent_name=LODGING)

        assert _drain() == []
        assert _names(scope) == [BOOK_HOTEL, BOOK_HOTEL]

    def test_fails_when_the_named_agent_called_none_of_them(self, asserter):
        asserter.called_any_tool(BOOK_FLIGHT, WEATHER, agent_name=LODGING)

        messages = _drain()
        assert len(messages) == 1
        assert BOOK_FLIGHT in messages[0] and WEATHER in messages[0]
        assert LODGING in messages[0]

    def test_counts_are_totals_across_the_named_tools(self, asserter):
        # Two book_flight calls and one weather call: three between them.
        asserter.called_any_tool(BOOK_FLIGHT, WEATHER, count=3)
        assert _drain() == []

        asserter.called_any_tool(BOOK_FLIGHT, WEATHER, max_count=2)
        messages = _drain()
        assert len(messages) == 1
        assert "3 times in total" in messages[0] and "at most 2" in messages[0]

    def test_count_message_names_the_calling_agent(self, asserter):
        asserter.called_any_tool(BOOK_HOTEL, agent_name=LODGING, min_count=3)

        messages = _drain()
        assert len(messages) == 1
        assert LODGING in messages[0]
        assert "2 times in total" in messages[0] and "at least 3" in messages[0]

    def test_chains_onto_an_output_check(self, asserter):
        asserter.called_any_tool(BOOK_FLIGHT, WEATHER).contains_any_output(
            "flight", "weather")

        assert _drain() == []


class TestDoesNotCallAnyAgent:

    def test_passes_when_no_named_agent_ran(self, asserter):
        asserter.does_not_call_any_agent("never_ran", "also_never_ran")

        assert _drain() == []

    def test_one_call_fails_it(self, asserter):
        asserter.does_not_call_any_agent("never_ran", LODGING)

        assert len(_drain()) == 1

    def test_the_failure_names_what_ran_not_the_candidates(self, asserter):
        asserter.does_not_call_any_agent(AIR, LODGING, "never_ran")

        messages = _drain()
        assert AIR in messages[0] and LODGING in messages[0]
        assert "never_ran" not in messages[0]

    def test_a_list_argument_is_accepted(self, asserter):
        asserter.does_not_call_any_agent(["never_ran", "also_never_ran"])

        assert _drain() == []

    def test_does_not_narrow_the_context(self, asserter):
        """A negative selects nothing, so every span stays in scope for what follows."""
        scope = asserter.does_not_call_any_agent("never_ran")

        assert _drain() == []
        assert list(scope._filtered_spans) == list(asserter.validator.spans)

    def test_custom_message_replaces_the_default(self, asserter):
        asserter.does_not_call_any_agent(LODGING, message="FORBIDDEN ROUTE TAKEN")

        assert _drain() == ["FORBIDDEN ROUTE TAKEN"]


class TestDoesNotCallAnyTool:

    def test_passes_when_no_named_tool_ran(self, asserter):
        asserter.does_not_call_any_tool("never_ran", "also_never_ran")

        assert _drain() == []

    def test_one_call_fails_it(self, asserter):
        asserter.does_not_call_any_tool("never_ran", BOOK_HOTEL)

        messages = _drain()
        assert len(messages) == 1
        assert BOOK_HOTEL in messages[0]

    def test_agent_name_restricts_what_counts_as_a_call(self, asserter):
        """book_flight ran, but not by the lodging agent, so this holds."""
        asserter.does_not_call_any_tool(BOOK_FLIGHT, WEATHER, agent_name=LODGING)

        assert _drain() == []

    def test_the_failure_names_the_calling_agent(self, asserter):
        asserter.does_not_call_any_tool(BOOK_FLIGHT, BOOK_HOTEL, agent_name=LODGING)

        messages = _drain()
        assert len(messages) == 1
        assert BOOK_HOTEL in messages[0] and LODGING in messages[0]

    def test_a_list_argument_is_accepted(self, asserter):
        asserter.does_not_call_any_tool([BOOK_FLIGHT, BOOK_HOTEL])

        assert len(_drain()) == 1

    def test_narrowed_context_limits_it(self, asserter):
        """After a selector, the negative only sees the spans still in scope.

        book_flight ran in this trace, so this passes only because the hotel
        selector took it out of scope first.
        """
        asserter.called_any_tool(BOOK_HOTEL).does_not_call_any_tool(BOOK_FLIGHT)

        assert _drain() == []

    def test_the_narrowed_spans_still_fail_their_own_name(self, asserter):
        """The other half of the pair: what IS in scope is still caught."""
        asserter.called_any_tool(BOOK_HOTEL).does_not_call_any_tool(BOOK_HOTEL)

        assert len(_drain()) == 1


class TestArgumentValidation:

    def test_no_names_is_a_usage_error(self, asserter):
        with pytest.raises(ValueError, match="At least one agent name is required"):
            asserter.called_any_agent()
        with pytest.raises(ValueError, match="At least one tool name is required"):
            asserter.called_any_tool([])
        with pytest.raises(ValueError, match="At least one agent name is required"):
            asserter.does_not_call_any_agent()
        with pytest.raises(ValueError, match="At least one tool name is required"):
            asserter.does_not_call_any_tool([])

    def test_non_string_names_are_a_usage_error(self, asserter):
        with pytest.raises(ValueError, match="agent names must be strings"):
            asserter.called_any_agent(AIR, 7)

    def test_count_with_min_or_max_is_a_usage_error(self, asserter):
        with pytest.raises(ValueError, match="Cannot specify both"):
            asserter.called_any_agent(AIR, count=1, min_count=1)
        with pytest.raises(ValueError, match="Cannot specify both"):
            asserter.called_any_tool(BOOK_FLIGHT, count=1, max_count=2)

    def test_testcase_points_at_the_all_of_selector(self, asserter):
        with pytest.raises(ValueError, match="Use called_agent"):
            asserter.called_any_agent(AIR, testcase={"agents": {AIR: {}}})
        with pytest.raises(ValueError, match="Use called_tool"):
            asserter.called_any_tool(BOOK_FLIGHT,
                                     testcase={"tools": {BOOK_FLIGHT: {}}})
