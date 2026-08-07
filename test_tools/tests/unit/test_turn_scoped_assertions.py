"""Scoping fluent assertions to a specific turn.

Every `run_agent_async(..., turn_id=...)` call runs inside its own turn scope,
so each span that run produces carries a `scope.turn_id` attribute. Two calls on
the same asserter therefore leave one pool of spans that is still separable by
turn: `where(attribute={"scope.turn_id": ...})` narrows the chain to a single
turn, and everything downstream of it -- `called_tool`, `does_not_call_tool`,
`contains_input`, ... -- sees only that turn's spans.

The framework integration tests need a live LLM, so these use a fake runner that
replays a recorded LangGraph trace: the flight-booking tool span on the first
call, the hotel-booking tool span on the second. The turn tagging itself is not
faked -- the runner reads the same `get_scopes()` that
`SpanHandler.set_default_monocle_attributes` reads when instrumentation stamps a
real span, so the scope under test is the one `run_agent_async` actually opened.
"""
import os

import pytest

from monocle_apptrace.instrumentation.common.utils import get_scopes
from monocle_test_tools import TraceAssertion
from monocle_test_tools.file_span_loader import JSONSpanLoader
from monocle_test_tools.runner.agent_runner import AgentRunner
import monocle_test_tools.validator as validator_module


SESSION = "turn_scope_session_test"
FLIGHT_TOOL = "okahu-demo-lg-tool_book_flight"
FLIGHT_AGENT = "okahu-demo-lg-agent-air_travel_assistant"
HOTEL_TOOL = "okahu-demo-lg-tool_book_hotel"
HOTEL_AGENT = "okahu-demo-lg-agent-lodging_assistant"


def _tool_spans(tool_name):
    """The recorded spans for one tool, replayed as a turn's worth of work."""
    here = os.path.dirname(os.path.abspath(__file__))
    trace = os.path.join(here, "traces", "trace2.json")
    return [span for span in JSONSpanLoader.from_json(trace)
            if span.attributes.get("entity.1.name") == tool_name]


class FakeTurnRunner(AgentRunner):
    """Replays one canned set of spans per call, tagged with the live scopes."""

    def __init__(self, validator, spans_per_call, outputs):
        self._validator = validator
        self._spans_per_call = spans_per_call
        self._outputs = outputs
        self._call = 0
        self.session_ids_seen = []

    async def run_agent_async(self, root_agent, *args, session_id=None):
        self.session_ids_seen.append(session_id)
        spans = self._spans_per_call[self._call]
        for span in spans:
            # What SpanHandler.set_default_monocle_attributes does for every
            # real span: copy whatever scopes are open onto the span. Here that
            # is the turn scope run_agent_async opened for this call.
            for scope_key, scope_value in get_scopes().items():
                span._attributes[f"scope.{scope_key}"] = scope_value
        self._validator.memory_exporter.export(spans)
        output = self._outputs[self._call]
        self._call += 1
        return output

    async def end_session(self, session_id=None) -> None:
        pass


@pytest.fixture
def two_turn_asserter(monkeypatch, monocle_trace_asserter):
    """An asserter whose fake runner books a flight, then a hotel."""
    fake = FakeTurnRunner(
        monocle_trace_asserter.validator,
        [_tool_spans(FLIGHT_TOOL), _tool_spans(HOTEL_TOOL)],
        ["Flight booked", "Hotel booked"],
    )
    monkeypatch.setattr(validator_module, "get_agent_runner", lambda agent_type: fake)
    return monocle_trace_asserter


@pytest.mark.asyncio
async def test_assertions_scoped_to_a_specific_turn(two_turn_asserter):
    """Two runs, one turn id each -- every assertion below names its turn."""
    await two_turn_asserter.run_agent_async(
        None, "google_adk", "Book me a flight to Mumbai for 26th Nov 2025.",
        session_id=SESSION, turn_id="flight")
    await two_turn_asserter.run_agent_async(
        None, "google_adk", "Now book me a hotel there.",
        session_id=SESSION, turn_id="hotel")

    # The flight was booked in the "flight" turn, the hotel in the "hotel" turn.
    two_turn_asserter \
        .where(attribute={"scope.turn_id": "flight"}) \
        .called_tool(FLIGHT_TOOL, FLIGHT_AGENT)
    two_turn_asserter \
        .where(attribute={"scope.turn_id": "hotel"}) \
        .called_tool(HOTEL_TOOL, HOTEL_AGENT)

    # ...and neither leaked into the other turn. This is what turn scoping buys:
    # unscoped, both tools were called, so neither does_not_call_tool would hold.
    two_turn_asserter \
        .where(attribute={"scope.turn_id": "flight"}) \
        .does_not_call_tool(HOTEL_TOOL)
    two_turn_asserter \
        .where(attribute={"scope.turn_id": "hotel"}) \
        .does_not_call_tool(FLIGHT_TOOL)

    # Unscoped, the asserter still sees the whole session: both turns' spans.
    two_turn_asserter.called_tool(FLIGHT_TOOL)
    two_turn_asserter.called_tool(HOTEL_TOOL)

    assert not two_turn_asserter.has_assertions(), two_turn_asserter.get_assertion_messages()


@pytest.mark.asyncio
async def test_assertion_against_the_wrong_turn_fails(two_turn_asserter):
    """The same assertion passes for its own turn and fails for the other one."""
    await two_turn_asserter.run_agent_async(
        None, "google_adk", "Book me a flight to Mumbai for 26th Nov 2025.",
        session_id=SESSION, turn_id="flight")
    await two_turn_asserter.run_agent_async(
        None, "google_adk", "Now book me a hotel there.",
        session_id=SESSION, turn_id="hotel")

    # See the try/finally note in test_check_eval_template_path.py: this test
    # deliberately leaves a recorded assertion behind, and pytest_plugin.py's
    # makereport hook fails any test whose asserter still holds one.
    try:
        result = two_turn_asserter \
            .where(attribute={"scope.turn_id": "hotel"}) \
            .called_tool(FLIGHT_TOOL)

        assert result.has_assertions()
        assert FLIGHT_TOOL in result.get_assertion_messages()
    finally:
        TraceAssertion._assertion_errors = []


@pytest.mark.asyncio
async def test_turn_id_is_on_every_span_of_its_turn(two_turn_asserter):
    """The tag lives on the spans, so it is assertable like any attribute."""
    await two_turn_asserter.run_agent_async(
        None, "google_adk", "Book me a flight to Mumbai for 26th Nov 2025.",
        session_id=SESSION, turn_id="flight")
    await two_turn_asserter.run_agent_async(
        None, "google_adk", "Now book me a hotel there.",
        session_id=SESSION, turn_id="hotel")

    turn_ids = {span.attributes.get("scope.turn_id")
                for span in two_turn_asserter.validator.spans}
    assert turn_ids == {"flight", "hotel"}

    two_turn_asserter.has_attribute("scope.turn_id", "flight")
    two_turn_asserter.has_attribute("scope.turn_id", "hotel")
    assert not two_turn_asserter.has_assertions(), two_turn_asserter.get_assertion_messages()


if __name__ == "__main__":
    pytest.main([__file__])
