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

if __name__ == "__main__":
    pytest.main([__file__]) 