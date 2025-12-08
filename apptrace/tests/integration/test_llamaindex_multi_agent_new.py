import logging
import os
import signal
import subprocess
import threading
import time

import pytest
import uvicorn
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp import aget_tools_from_mcp_url
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, Field

from integration.servers.mcp.weather_server import app as weather_app

logger = logging.getLogger(__name__)



# Global variables to track server processes
weather_server_process = None
a2a_server_process = None


def start_weather_server():
    """Start the weather MCP server on port 8001."""

    def run_server():
        uvicorn.run(weather_app, host="127.0.0.1", port=8001, log_level="error")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Wait for server to start
    return server_thread


def start_a2a_server():
    """Start the A2A Currency server on port 10000."""
    import os
    import sys

    # Get the path to the A2A server script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    a2a_server_path = os.path.join(
        current_dir, "servers", "a2a_langgraph", "app", "__main__.py"
    )

    # Start the A2A server as a subprocess
    process = subprocess.Popen(
        [sys.executable, a2a_server_path, "--host", "localhost", "--port", "10000"],
        stdout=subprocess.DEVNULL,  # Suppress output
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if os.name != "nt" else None,  # For proper cleanup
    )
    time.sleep(3)  # Wait for server to start
    return process


def stop_a2a_server(process):
    """Stop the A2A server process."""
    if process and process.poll() is None:
        try:
            if os.name != "nt":
                # Unix-like systems
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                # Windows
                process.terminate()
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            if process.poll() is None:
                process.kill()


# pytest start servers once for the module
@pytest.fixture(scope="module")
def start_servers():
    """Start both weather and A2A servers for the test module."""
    global weather_server_process, a2a_server_process

    logger.info("Starting weather server...")
    weather_server_process = start_weather_server()

    logger.info("Starting A2A server...")
    a2a_server_process = start_a2a_server()

    yield

    # Cleanup
    logger.info("Stopping servers...")
    if a2a_server_process:
        stop_a2a_server(a2a_server_process)


@pytest.fixture(scope="module")
def setup(start_servers):
    memory_exporter = InMemorySpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        SimpleSpanProcessor(memory_exporter),
        BatchSpanProcessor(file_exporter),
    ]
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="llamaindex_agent_1",
            span_processors=span_processors,
        )
        yield memory_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name} for 50 USD."


def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return (
        f"Successfully booked a flight from {from_airport} to {to_airport} for 100 USD."
    )


class CurrencyConversionInput(BaseModel):
    message: str = Field(
        description="The currencies in prompt like : 'Give exchange rate for USD to EUR'"
    )


def currency_conversion_tool(message: str) -> str:
   return f"Currency conversion for: {message}. Exchange rate USD to EUR is 0.85 (simulated)"


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


async def setup_agents():
    """Setup multi-agent workflow with MCP tools integration."""

    # Set up MCP client for weather tools
    async def get_mcp_tools():
        """Get tools from the MCP weather server."""
        try:
            weather_tools = await aget_tools_from_mcp_url(
                "http://localhost:8001/weather/mcp/"
            )
            return weather_tools
        except Exception as e:
            logger.warning(f"Failed to get MCP tools: {e}")
            return []

    # Get MCP weather tools
    weather_tools = await get_mcp_tools()

    # Create additional tools
    currency_tool = FunctionTool.from_defaults(
        fn=currency_conversion_tool,
        name="CurrencyConversion",
        description="Gives currency conversion rate",
    )

    multiply_tool = FunctionTool.from_defaults(
        fn=multiply, name="multiply", description="Multiply two numbers."
    )

    # Additional tools for supervisor
    supervisor_tools = weather_tools + [currency_tool, multiply_tool]

    llm = OpenAI(model="gpt-4o", temperature=0.0, max_tokens=5000)

    #     flight_tool = FunctionTool.from_defaults(
    #         fn=book_flight,
    #         name="book_flight",
    #         description="Books a flight from one airport to another."
    #     )
    #     flight_agent = FunctionAgent(name="flight_booking_agent", tools=[flight_tool], llm=llm,
    #                             system_prompt="You are a flight booking agent who books flights as per the request. Once you complete the task, you handoff to the coordinator agent only.",
    #                             description="Flight booking agent")

    #     hotel_tool = FunctionTool.from_defaults(
    #         fn=book_hotel,
    #         name="book_hotel",
    #         description="Books a hotel stay."
    #     )
    #     hotel_agent = FunctionAgent(name="hotel_booking_agent", tools=[hotel_tool], llm=llm,
    #                             system_prompt="You are a hotel booking agent who books hotels as per the request. Once you complete the task, you handoff to the coordinator agent only.",
    #                             description="Hotel booking agent")

    #     coordinator = FunctionAgent(name="coordinator", tools=[], llm=llm,
    #                             system_prompt=
    #                             """You are a coordinator agent who manages the flight and hotel booking agents. Separate hotel booking and flight booking tasks clearly from the input query.
    #                             Delegate only hotel booking to the hotel booking agent and only flight booking to the flight booking agent.
    #                             Once they complete their tasks, you collect their responses and provide consolidated response to the user.""",
    #                             description="Travel booking coordinator agent",
    #                             can_handoff_to=["flight_booking_agent", "hotel_booking_agent"])

    #     agent_workflow = AgentWorkflow(
    #         agents=[coordinator, flight_agent, hotel_agent],
    #         root_agent=coordinator.name
    #     )
    #     return agent_workflow

    # async def run_agent():
    #     """Test multi-agent interaction with flight and hotel booking."""

    #     agent_workflow = setup_agents()
    #     resp = await agent_workflow.run(user_msg="book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel")
    #     logger.info(resp)

    # Flight booking agent
    flight_tool = FunctionTool.from_defaults(
        fn=book_flight,
        name="book_flight",
        description="Books a flight from one airport to another. Use it to directly book the flight",
    )
    flight_agent = FunctionAgent(
        name="flight_assistant",
        tools=[flight_tool],
        llm=llm,
        system_prompt="""You are a flight booking agent who books flights as per the request. 
        When you receive a flight booking request, immediately use the book_flight tool to complete the booking.
        After successfully booking the flight, handoff back to the supervisor agent with the booking details.""",
        description="Flight booking agent",
        can_handoff_to=["supervisor"],  # Can handoff to supervisor agent
    )

    # Hotel booking agent
    hotel_tool = FunctionTool.from_defaults(
        fn=book_hotel,
        name="book_hotel",
        description="Books a hotel stay. Use it to directly book the hotel.",
    )
    hotel_agent = FunctionAgent(
        name="hotel_assistant",
        tools=[hotel_tool],
        llm=llm,
        system_prompt="""You are a hotel booking agent who books hotels as per the request.
        When you receive a hotel booking request, immediately use the book_hotel tool to complete the booking.
        After successfully booking the hotel, handoff back to the supervisor agent with the booking details.""",
        description="Hotel booking agent",
        can_handoff_to=["supervisor"],  # Can handoff to supervisor agent
    )

    # Supervisor agent with additional tools
    supervisor = FunctionAgent(
        name="supervisor",
        tools=supervisor_tools,
        llm=llm,
        system_prompt="""You are a coordinator agent who manages flight and hotel booking agents. 
                         
                         For each user request:
                         1. First delegate flight booking to the flight_assistant agent
                         2. After flight booking is complete, delegate hotel booking to the hotel_assistant agent  
                         3. Once both bookings are complete, use currency conversion tools if needed
                         4. Provide a consolidated response with all booking details and costs
                         
                         Always ensure both agents complete their tasks before providing the final response.""",
        description="Travel booking supervisor agent",
        can_handoff_to=["flight_assistant", "hotel_assistant"],
    )

    agent_workflow = AgentWorkflow(
        handoff_prompt="""As soon as you have figured out the requirements, 
        If you need to delegate the task to another agent, then delegate the task to that agent.
        For eg if you need to book a flight, then delegate the task to flight agent.
        If you need to book a hotel, then delegate the task to hotel agent.
        If you can book hotel or flight direclty, then do that and collect the response and then handoff to the supervisor agent.
{agent_info}
        """,
        #         handoff_output_prompt="""
        #     Agent {to_agent} is now handling the request.
        #     Check the previous chat history and continue responding to the user's request: {user_request}.
        # """,
        agents=[supervisor, flight_agent, hotel_agent],
        root_agent=supervisor.name,
    )
    return agent_workflow


async def run_async_agent():
    """Test async multi-agent interaction with more complex requirements."""
    agent_workflow = await setup_agents()
    resp = await agent_workflow.run(
        user_msg="Book a flight from BOS to JFK, and a hotel stay at McKittrick Hotel. Give me the cost in INR."
    )
    logger.info(resp)


@pytest.mark.asyncio
async def test_async_multi_agent(setup):
    """Test async multi-agent interaction with weather and currency tools."""
    await run_async_agent()
    verify_spans(memory_exporter=setup)


def verify_spans(memory_exporter=None):
    time.sleep(2)
    found_inference = found_agent = found_tool = False
    found_flight_agent = found_hotel_agent = found_supervisor_agent = False
    found_book_hotel_tool = found_book_flight_tool = False
    found_book_flight_delegation = found_book_hotel_delegation = False
    spans = memory_exporter.get_finished_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference"
            or span_attributes["span.type"] == "inference.framework"
        ):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4o"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4o"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            ### TODO: Handle streaming responses from OpenAI LLM called by LlamaIndex agentic workflow
            ##            assert "completion_tokens" in span_metadata.attributes
            ##            assert "prompt_tokens" in span_metadata.attributes
            ##            assert "total_tokens" in span_metadata.attributes
            found_inference = True

        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.invocation"
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.llamaindex"
            
            # Check for delegation via from_agent attribute
            if "entity.1.from_agent" in span_attributes:
                from_agent = span_attributes["entity.1.from_agent"]
                agent_name = span_attributes["entity.1.name"]
                if agent_name == "flight_assistant" and from_agent == "supervisor":
                    found_book_flight_delegation = True
                elif agent_name == "hotel_assistant" and from_agent == "supervisor":
                    found_book_hotel_delegation = True
            
            if span_attributes["entity.1.name"] == "flight_assistant":
                found_flight_agent = True
            elif span_attributes["entity.1.name"] == "hotel_assistant":
                found_hotel_agent = True
            elif span_attributes["entity.1.name"] == "supervisor":
                found_supervisor_agent = True
            found_agent = True

        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert "entity.2.type" in span_attributes
            assert "entity.2.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.llamaindex"
            if (
                span_attributes["entity.1.name"] == "book_flight"
                and span_attributes["entity.2.name"] == "flight_assistant"
            ):
                found_book_flight_tool = True
            elif (
                span_attributes["entity.1.name"] == "book_hotel"
                and span_attributes["entity.2.name"] == "hotel_assistant"
            ):
                found_book_hotel_tool = True
            found_tool = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    assert found_book_flight_delegation, "Book flight delegation span not found (check from_agent attribute)"
    assert found_book_hotel_delegation, "Book hotel delegation span not found (check from_agent attribute)"
    assert found_flight_agent, "Flight assistant agent span not found"
    assert found_hotel_agent, "Hotel assistant agent span not found"
    assert found_supervisor_agent, "Supervisor agent span not found"
    assert found_book_flight_tool, "Book flight tool span not found"
    assert found_book_hotel_tool, "Book hotel tool span not found"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
