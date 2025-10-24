import logging
import os
import threading
import time

import pytest
import uvicorn
from common.custom_exporter import CustomConsoleSpanExporter
from config.conftest import temporary_env_var
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from integration.servers.mcp.weather_server import app as weather_app

logger = logging.getLogger(__name__)


# Global variable to track weather server process
weather_server_process = None


def start_weather_server():
    """Start the weather MCP server on port 8001."""

    def run_server():
        uvicorn.run(weather_app, host="127.0.0.1", port=8001, log_level="error")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Wait for server to start
    return server_thread


@pytest.fixture(scope="module")
def start_weather_server_fixture():
    """Start weather server for the test module."""
    global weather_server_process
    
    logger.info("Starting weather server...")
    weather_server_process = start_weather_server()
    
    yield
    
    # Cleanup
    logger.info("Stopping weather server...")


@pytest.fixture(scope="module")
def setup(start_weather_server_fixture):
    
    memory_exporter = InMemorySpanExporter()
    custom_exporter = CustomConsoleSpanExporter()
    span_processors = [
        SimpleSpanProcessor(memory_exporter),
        BatchSpanProcessor(FileSpanExporter()),
        SimpleSpanProcessor(custom_exporter)
    ]
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="agents_sdk_dev_1",
            # monocle_exporters_list="file, okahu"
            span_processors=span_processors,
        )
        yield memory_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def book_hotel(hotel_name: str) -> str:
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name} for 50 USD."


def book_flight(from_airport: str, to_airport: str) -> str:
    """Book a flight"""
    return (
        f"Successfully booked a flight from {from_airport} to {to_airport} for 100 USD."
    )


# def get_weather(city: str) -> str:
#     """Get weather information for a city"""
#     return f"The weather in {city} is sunny and 75°F."


@pytest.mark.asyncio
async def test_agents_sdk_multi_agent(setup):
    """Test multi-agent interaction with handoffs and MCP servers."""
    try:
        from agents import Agent, Runner, function_tool
        from agents.mcp import MCPServerStreamableHttp

        # Create MCP server for weather
        weather_mcp_server = MCPServerStreamableHttp(
            params={"url": "http://localhost:8001/weather/mcp/"}
        )
        
        await weather_mcp_server.connect()

        # Create tools
        @function_tool
        def book_flight_tool(from_airport: str, to_airport: str) -> str:
            """Book a flight between airports."""
            return book_flight(from_airport, to_airport)

        @function_tool
        def book_hotel_tool(hotel_name: str) -> str:
            """Book a hotel reservation."""
            return book_hotel(hotel_name)

        # @function_tool
        # def get_weather_tool(city: str) -> str:
        #     """Get weather information."""
        #     return get_weather(city)

        # Create specialized agents with MCP servers
        flight_agent = Agent(
            name="Flight Agent",
            instructions="You are a flight booking specialist. Use the book_flight_tool to book flights and get_weather to check weather at destinations.",
            tools=[book_flight_tool],
            mcp_servers=[weather_mcp_server],
        )

        hotel_agent = Agent(
            name="Hotel Agent",
            instructions="You are a hotel booking specialist. Use the book_hotel_tool to book hotels and get_weather to check weather conditions.",
            tools=[book_hotel_tool],
            mcp_servers=[weather_mcp_server],
        )

        # Create a coordinator agent with handoffs and MCP servers
        coordinator = Agent(
            name="Travel Coordinator",
            instructions="You are a travel coordinator. Delegate flight bookings to the Flight Agent and hotel bookings to the Hotel Agent. Use weather information to make informed recommendations.",
            handoffs=[flight_agent, hotel_agent],
            mcp_servers=[weather_mcp_server],
        )

        # Test the multi-agent workflow with weather information
        result = await Runner.run(
            coordinator,
            "Please help me with my complete travel needs:Book a flight from NYC to LAX using the flight agent. Also check the weather in both cities. Please delegate to the appropriate specialized agents.",
        )

        logger.info(f"Multi-agent result: {result.final_output}")

        result = await Runner.run(
            coordinator,
            "Please help me with my complete travel needs: Book the Hilton hotel in Los Angeles using the hotel agent, Also check the weather in both cities. Please delegate to the appropriate specialized agents.",
        )

        logger.info(f"Multi-agent result: {result.final_output}")

        # Verify spans were created
        verify_multi_agent_spans(memory_exporter=setup)

    except ImportError:
        pytest.skip("OpenAI Agents SDK not available")

def verify_multi_agent_spans(memory_exporter=None):
    """Verify that multi-agent spans were created."""
    time.sleep(2)
    # Allow time for spans to be processed

    found_agent = found_tool = found_delegation = found_mcp = False
    found_flight_delegation = found_hotel_delegation = False
    agent_names = set()
    tool_names = set()
    mcp_operations = set()

    spans = memory_exporter.get_finished_spans()

    for span in spans:
        span_attributes = span.attributes

        # Check for agent spans
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.invocation"
        ):
            assert span_attributes["entity.1.type"] == "agent.openai_agents"
            assert "entity.1.name" in span_attributes
            agent_names.add(span_attributes["entity.1.name"])
            found_agent = True

        # Check for subtype in inference
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "inference"
        ):
            assert "span.subtype" in span.attributes, "Expected span.subtype attribute to be present"
            assert span.attributes.get("span.subtype") in ["turn_end", "tool_call", "delegation"]

        # Check for tool spans
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            if "is_mcp" in span_attributes and span_attributes["is_mcp"]:
                assert span_attributes["entity.1.type"] == "tool.mcp"
            else:
                assert span_attributes["entity.1.type"] == "tool.openai_agents"
            assert "entity.1.name" in span_attributes
            tool_names.add(span_attributes["entity.1.name"])
            found_tool = True

        # Check for delegation spans
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.delegation"
        ):
            assert span_attributes["entity.1.type"] == "agent.openai_agents"
            assert "entity.1.from_agent" in span_attributes
            assert "entity.1.to_agent" in span_attributes
            found_delegation = True
            
            # Check for specific agent delegations
            if span_attributes["entity.1.to_agent"] == "Flight Agent":
                found_flight_delegation = True
            elif span_attributes["entity.1.to_agent"] == "Hotel Agent":
                found_hotel_delegation = True

        # Check for MCP-related spans
        if (
            "span.type" in span_attributes
            and "mcp" in span_attributes["span.type"]
        ):
            mcp_operations.add(span_attributes["span.type"])
            found_mcp = True

    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    assert found_delegation, "Delegation span not found"
    assert found_flight_delegation, "Flight Agent delegation span not found"
    assert found_hotel_delegation, "Hotel Agent delegation span not found"
    assert found_mcp, "MCP operation span not found"
    # Note: Delegation might not always occur depending on the model's decisions
    # Note: MCP spans might not always occur depending on whether MCP tools are called

    logger.info(f"Found agents: {agent_names}")
    logger.info(f"Found tools: {tool_names}")
    if mcp_operations:
        logger.info(f"Found MCP operations: {mcp_operations}")


@pytest.mark.asyncio
async def test_agents_sdk_mcp_server(setup):
    """Test OpenAI Agents SDK with MCP server integration."""
    try:
        from agents import Agent, Runner
        from agents.mcp import MCPServerStreamableHttp

        # Create MCP server for weather
        weather_mcp_server = MCPServerStreamableHttp(
            params={"url": "http://localhost:8001/weather/mcp/"}
        )
        await weather_mcp_server.connect()

        # Create an agent that uses only MCP tools
        weather_agent = Agent(
            name="Weather Assistant",
            instructions="You are a weather information specialist. Use the available weather tools to provide accurate weather information for cities.",
            mcp_servers=[weather_mcp_server],
        )

        # Test the agent with weather queries
        result = await Runner.run(
            weather_agent,
            "What's the weather like in New York, London, and Tokyo? Please provide the temperature for each city.",
        )

        logger.info(f"Weather agent result: {result.final_output}")

        # Verify spans were created
        verify_mcp_spans(memory_exporter=setup)

    except ImportError:
        pytest.skip("OpenAI Agents SDK not available")

@pytest.mark.asyncio
async def test_invalid_api_key_error_code_in_span(setup):
    """Test that passing an invalid API key results in error_code in the span."""
    # Simulate invalid API key by setting an obviously wrong value
    setup.clear()
    
    with temporary_env_var("OPENAI_API_KEY", "INVALID_API_KEY"):
        try:
            try:
                from agents import Agent, Runner
                from agents.mcp import MCPServerStreamableHttp

                # Create MCP server for weather
                weather_mcp_server = MCPServerStreamableHttp(
                    params={"url": "http://localhost:8001/weather/mcp/"}
                )
                await weather_mcp_server.connect()

                # Create an agent that uses only MCP tools
                weather_agent = Agent(
                    name="Weather Assistant",
                    instructions="You are a weather information specialist. Use the available weather tools to provide accurate weather information for cities.",
                    mcp_servers=[weather_mcp_server],
                )

                # Test the agent with weather queries
                result = await Runner.run(
                    weather_agent,
                    "What's the weather like in New York, London, and Tokyo? Please provide the temperature for each city.",
                )

                logger.info(f"Weather agent result: {result.final_output}")

                # Verify spans were created
                verify_mcp_spans(memory_exporter=setup)

            except ImportError:
                pytest.skip("OpenAI Agents SDK not available")

        except Exception:
            spans = setup.get_finished_spans()
            for span in spans:
                span_attributes = span.attributes
                if (
                        "span.type" in span_attributes
                        and span_attributes["span.type"] == "agentic.invocation"
                        and "entity.1.name" in span_attributes
                ):
                    span_input, span_output, _, _ = span.events
                    assert "error_code" in span_output.attributes
                    assert span_output.attributes["error_code"] == "invalid_api_key"


def verify_mcp_spans(memory_exporter=None):
    """Verify that MCP-related spans were created."""
    time.sleep(2)
    # Allow time for spans to be processed

    found_agent = found_mcp_tool = found_mcp_list = False
    agent_names = set()
    mcp_tool_names = set()

    spans = memory_exporter.get_finished_spans()

    for span in spans:
        span_attributes = span.attributes

        # Check for agent spans
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.invocation"
        ):
            assert span_attributes["entity.1.type"] == "agent.openai_agents"
            assert "entity.1.name" in span_attributes
            agent_names.add(span_attributes["entity.1.name"])
            found_agent = True

        # Check for subtype in inference
        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "inference"
        ):
            assert "span.subtype" in span.attributes, "Expected span.subtype attribute to be present"
            assert span.attributes.get("span.subtype") in ["turn_end", "tool_call", "delegation"]

        # Check for MCP tool invocation spans
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.tool.invocation"
            and "entity.1.name" in span_attributes
            and span_attributes["entity.1.name"] == "get_weather"
        ):
            mcp_tool_names.add(span_attributes["entity.1.name"])
            found_mcp_tool = True

        # Check for MCP list tools spans (if they exist)
        if (
            "span.type" in span_attributes
            and "mcp" in span_attributes.get("span.type", "").lower()
        ):
            found_mcp_list = True

    assert found_agent, "Agent span not found"
    # Note: MCP tool spans might not always occur depending on whether MCP tools are called

    logger.info(f"Found agents: {agent_names}")
    if mcp_tool_names:
        logger.info(f"Found MCP tools: {mcp_tool_names}")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
