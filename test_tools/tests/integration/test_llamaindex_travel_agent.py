from asyncio import sleep
import logging
import pytest 
import logging
import uvicorn
import threading
import time
from dotenv import load_dotenv

from test_common.llamaindex_travel_agent import (
    setup_agents,
    run_react_agent_achat,
    run_react_agent_aquery,
    run_react_agent_chat,
    run_query_engine,
    run_query_engine_async
)
from monocle_test_tools import TestCase, MonocleValidator
from test_common.weather_mcp_server import app as weather_app
logger = logging.getLogger(__name__)

load_dotenv()


def start_weather_server():
    """Start the weather MCP server on port 8001."""

    def run_server():
        uvicorn.run(weather_app, host="127.0.0.1", port=8001, log_level="error")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Wait for server to start
    return server_thread

logger.info("Starting weather server...")
weather_server_process = start_weather_server()


agent_test_cases:list[TestCase] = [
    {
        "test_input": ["Book a flight from San Jose to Boston and a book hotel stay at Hyatt Hotel, and tell the weather in Boston."],
        "test_output": "Here are your booking details and weather information for Boston:\n\n- **Flight**: Successfully booked fro...n.\n- **Weather in Boston**: The current temperature is 69°F.",
        "comparer": "similarity",
    },
    {
        "test_input": ["Book a flight from San Francisco to Mumbai. Book a two queen room at Marriot Intercontinental at Central Mumbai and tell me the weather of Mumbai."],
        "test_spans": [

            {
            "span_type": "agentic.invocation",
            "entities": [
                {"type": "agent", "name": "lmx_coordinator_05"}
                ]
            },
            {
            "span_type": "agentic.invocation",
            "entities": [
                {"type": "agent", "name": "lmx_flight_booking_agent_05"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "lmx_book_flight_tool_05"},
                    {"type": "agent", "name": "lmx_flight_booking_agent_05"}
                ]
            },
            {
            "span_type": "agentic.invocation",
            "entities": [
                {"type": "agent", "name": "lmx_hotel_booking_agent_05"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "lmx_book_hotel_tool_05"},
                     {"type": "agent", "name": "lmx_hotel_booking_agent_05"}
                ]
            }
        ]
    }
]

@MonocleValidator().monocle_testcase(agent_test_cases)
async def test_run_agents(my_test_case: TestCase):
    agent_workflow = await setup_agents()
    await MonocleValidator().test_agent_async(agent_workflow, "llamaindex", my_test_case)
    await sleep(10)


# Test cases for QueryEngine with aquery method
query_engine_async_test_cases: list[TestCase] = [
    {
        "test_input": ["What are the hotel booking hours?"],
        "test_output": "Hotel reservations can be made online or by phone",
        "comparer": "similarity",
    }
]

@MonocleValidator().monocle_testcase(query_engine_async_test_cases)
async def test_query_engine_async(my_test_case: TestCase):
    """Test QueryEngine using aquery method (async query)."""
    await MonocleValidator().test_agent_async(run_query_engine_async, "llamaindex", my_test_case)
    await sleep(5)


# Test cases for QueryEngine with aquery method
react_agent_achat_test_cases: list[TestCase] = [
    {
        "test_input": ["Book a flight from San Jose to Boston and a book hotel stay at Hyatt Hotel."],
        "test_output": "A flight from San Jose to Boston has been booked, along with a stay at Hyatt Hotel.",
        "comparer": "similarity",
    }
]

@MonocleValidator().monocle_testcase(react_agent_achat_test_cases)
async def test_react_agent_achat(my_test_case: TestCase):
    """Test ReActAgent using aquery method (async query)."""
    await MonocleValidator().test_agent_async(run_react_agent_achat, "llamaindex", my_test_case)
    await sleep(5)


# Test cases for ReActAgent with aquery method
react_agent_aquery_test_cases: list[TestCase] = [
    {
        "test_input": ["Book a flight from San Jose to Boston and a book hotel stay at Hyatt Hotel."],
        "test_output": "A flight from San Jose to Boston has been booked, along with a stay at Hyatt Hotel.",
        "comparer": "similarity",
    }
]

@MonocleValidator().monocle_testcase(react_agent_aquery_test_cases)
async def test_react_agent_aquery(my_test_case: TestCase):
    """Test ReActAgent using aquery method (async query)."""
    await MonocleValidator().test_agent_async(run_react_agent_aquery, "llamaindex", my_test_case)
    await sleep(5)


# Test cases for ReActAgent with synchronous chat method
react_chat_test_cases: list[TestCase] = [
    {
        "test_input": ["Book a flight from Boston to Seattle"],
        "test_spans": [
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "lmx_book_flight_tool_sync"}
                ]
            }
        ]
    }
]

@MonocleValidator().monocle_testcase(react_chat_test_cases)
async def test_react_agent_chat(my_test_case: TestCase):
    """Test ReActAgent using synchronous chat method."""
    await MonocleValidator().test_agent_async(run_react_agent_chat, "llamaindex", my_test_case)
    await sleep(5)


# Test cases for QueryEngine with synchronous query method
query_engine_test_cases: list[TestCase] = [
    {
        "test_input": ["When is flight booking service available?"],
        "test_output": "Flight booking service is available from 9 AM to 5 PM",
        "comparer": "similarity",
    }
]

@MonocleValidator().monocle_testcase(query_engine_test_cases)
async def test_query_engine(my_test_case: TestCase):
    """Test QueryEngine using synchronous query method."""
    await MonocleValidator().test_agent_async(run_query_engine, "llamaindex", my_test_case)
    await sleep(5)


if __name__ == "__main__":
    pytest.main([__file__]) 