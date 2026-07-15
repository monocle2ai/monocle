import json
import os

import pytest
from dotenv import load_dotenv

from monocle_test_tools import TraceAssertion
from test_common.adk_travel_agent import remote_root_agent
from test_common.fastapi_helper import get_url, start_fastapi, stop_fastapi, wait_for_server

AGENT_ASK_API = "/api/v1/ask_agent"
load_dotenv()
os.environ["MONOCLE_EXPORTER"] = "okahu"

@pytest.fixture(scope="module", autouse=True)
def fastapi_server():
    """Start the local FastAPI agent service in a separate process, wait for it
    to be healthy, then tear it down after the module's tests complete."""
    start_fastapi()
    try:
        wait_for_server(timeout=30.0, interval=1.0)
    except Exception:
        stop_fastapi()
        raise
    server_url = get_url()
    # If AGENT_ENDPOINT isn't explicitly set, point it at the local server
    # so tests targeting a remote agent fall back to the in-process one.
    if not os.getenv("AGENT_ENDPOINT"):
        os.environ["AGENT_ENDPOINT"] = server_url
    try:
        yield server_url
    finally:
        stop_fastapi()


def _agent_endpoint() -> str:
    return os.getenv("AGENT_ENDPOINT", get_url())


# Inventory & Timeframe Coverage
def test_agent_remote(monocle_trace_asserter: TraceAssertion):
    """Agent efficiently fetches specific trace when trace_id provided"""
    headers = {"Content-Type": "application/json"}
    method = "POST"
    endpoint = _agent_endpoint() + AGENT_ASK_API

    # Call remote agent
    data = json.dumps({"query": "Book a flight from New York to San Francisco for 2nd May 2028"})
    monocle_trace_asserter.run_agent(
        agent=endpoint,
        agent_type="http_with_okahu",
        data=data,
        headers=headers,
        method=method,
    )

    # assert expected behavior
    monocle_trace_asserter.contains_output("San Francisco").contains_output("New York")
    monocle_trace_asserter.called_tool(tool_name="adk_book_flight", agent_name="adk_flight_booking_agent")
    monocle_trace_asserter.with_evaluation("okahu").check_eval("sentiment", expected="positive")

    # Call remote agent
    data = json.dumps({"query": "Book a room at Hyatt in San Francisco near airport for tomorrow night for one night"})
    monocle_trace_asserter.run_agent(
        agent=endpoint,
        agent_type="http_with_okahu",
        data=data,
        headers=headers,
        method=method,
    )

    # assert expected behavior
    monocle_trace_asserter.contains_output("San Francisco").contains_output("successfully booked")
    monocle_trace_asserter.called_tool(tool_name="adk_book_hotel", agent_name="adk_hotel_booking_agent")

if __name__ == "__main__":
    pytest.main()
