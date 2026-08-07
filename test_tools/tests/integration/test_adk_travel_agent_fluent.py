from asyncio import sleep
import pytest

from monocle_test_tools import TraceAssertion
from test_common.adk_travel_agent import root_agent, root_agent_parallel

@pytest.mark.asyncio
async def test_tool_invocation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", 
                        "Book a flight from San Francisco to Mumbai for 26th Nov 2025")
    monocle_trace_asserter.called_tool("adk_book_flight_5","adk_flight_booking_agent_5").contains_input("Mumbai")
    monocle_trace_asserter.under_token_limit(1000000)

@pytest.mark.asyncio
async def test_agent_invocation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5").contains_input("Book a flight from San Jose to Seattle")

# PARALLEL EXECUTION TESTS with fluent API
@pytest.mark.asyncio
async def test_parallel_agent_execution(monocle_trace_asserter):
    """Test that parallel agent executes flight and hotel booking concurrently."""
    await monocle_trace_asserter.run_agent_async(root_agent_parallel, "google_adk",
                        "Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a hotel at Marriott in Mumbai for 27th Nov 2025 for 4 nights.")
    
    # Verify all agents were called
    monocle_trace_asserter.called_agent("adk_parallel_booking_coordinator_5")
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5")
    monocle_trace_asserter.called_agent("adk_hotel_booking_agent_5")
    monocle_trace_asserter.called_agent("adk_trip_summary_agent_5")
    
    # Verify tools were invoked
    monocle_trace_asserter.called_tool("adk_book_flight_5", "adk_flight_booking_agent_5")
    monocle_trace_asserter.called_tool("adk_book_hotel_5", "adk_hotel_booking_agent_5")

@pytest.mark.asyncio
async def test_parallel_agent_has_execution_id(monocle_trace_asserter):
    """Verify that scope.agentic.executionId is present for ParallelAgent."""
    await monocle_trace_asserter.run_agent_async(root_agent_parallel, "google_adk",
                        "Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a hotel at Marriott in Mumbai.")
    
    # The parallel coordinator agent should have scope.agentic.executionId
    # This verifies parallel execution detection is working
    monocle_trace_asserter.called_agent("adk_parallel_booking_coordinator_5")
    
    # Verify correct output from summary agent
    monocle_trace_asserter.contains_output("flight")
    monocle_trace_asserter.contains_output("hotel")

# TURN-SCOPED ASSERTIONS with fluent API
@pytest.mark.asyncio
async def test_assertions_scoped_to_a_specific_turn(monocle_trace_asserter):
    """Two runs on one asserter, each tagged with its own turn id.

    Passing `turn_id` runs the agent inside a turn scope, so every span that run
    produces carries `scope.turn_id`. Both runs' spans land in the same asserter,
    and `where(attribute={"scope.turn_id": ...})` narrows the chain to one turn --
    without it, `called_tool("adk_book_flight_5").contains_input("Mumbai")` would
    be checking a tool span that could have come from either booking.

    Note these are two independent runs sharing only a turn tag: each
    `run_agent_async` builds its own runner, so the agent carries no memory from
    the first booking into the second (that is what `MultiTurnTestCase` /
    `run_multi_turn_agent_async` are for, and they assign turn ids the same way).
    Each turn's input is therefore self-contained.
    """
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Francisco to Mumbai for 26th Nov 2025.",
                        session_id="adk_turn_scope_session", turn_id="mumbai")
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.",
                        session_id="adk_turn_scope_session", turn_id="seattle")

    # Each turn booked its own destination...
    monocle_trace_asserter \
        .where(attribute={"scope.turn_id": "mumbai"}) \
        .called_tool("adk_book_flight_5", "adk_flight_booking_agent_5") \
        .contains_input("Mumbai")
    monocle_trace_asserter \
        .where(attribute={"scope.turn_id": "seattle"}) \
        .called_tool("adk_book_flight_5", "adk_flight_booking_agent_5") \
        .contains_input("Seattle")

    # ...and neither turn's booking leaked into the other.
    monocle_trace_asserter \
        .where(attribute={"scope.turn_id": "mumbai"}) \
        .called_tool("adk_book_flight_5") \
        .does_not_contain_input("Seattle")
    monocle_trace_asserter \
        .where(attribute={"scope.turn_id": "seattle"}) \
        .called_tool("adk_book_flight_5") \
        .does_not_contain_input("Mumbai")

if __name__ == "__main__":
    pytest.main([__file__])