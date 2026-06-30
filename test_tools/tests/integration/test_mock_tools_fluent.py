
import pytest

from monocle_test_tools import MockTool, TraceAssertion
from test_common.adk_travel_agent import remote_root_agent

@pytest.mark.asyncio
async def test_agent_local_with_mock(monocle_trace_asserter: TraceAssertion):
    mock_flight_tool = MockTool(
        name="adk_book_flight",
        type="tool.adk",
        response={"status": "success", "message": "Flight booked from New York to San Francisco."},
    )
    await monocle_trace_asserter.with_mock_tool(mock_flight_tool).run_agent_async(
        remote_root_agent, "google_adk", "Book a flight from New York to San Francisco for 2nd May 2028"
    )

    # assert expected behavior
    monocle_trace_asserter.contains_output("San Francisco").contains_output("New York")
    monocle_trace_asserter.called_tool(tool_name="adk_book_flight", agent_name="adk_flight_booking_agent")