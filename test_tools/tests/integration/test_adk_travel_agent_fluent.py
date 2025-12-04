from asyncio import sleep
import pytest

from monocle_test_tools import TraceAssertion
from test_common.adk_travel_agent import root_agent

@pytest.fixture(scope="function")
def trace_asserter():
    yield TraceAssertion.get_trace_asserter()

@pytest.mark.asyncio
async def test_tool_invocation(trace_asserter):
    await trace_asserter.run_agent_async(root_agent, "google_adk", 
                        "Book a flight from San Francisco to Mumbai for 26th Nov 2025")
    trace_asserter.called_tool("adk_book_flight_5","adk_flight_booking_agent_5").contains_input("Mumbai")

@pytest.mark.asyncio
async def test_agent_invocation(trace_asserter):
    await trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")
    trace_asserter.called_agent("adk_flight_booking_agent_5").contains_input("Book a flight from San Jose to Seattle")

if __name__ == "__main__":
    pytest.main([__file__]) 