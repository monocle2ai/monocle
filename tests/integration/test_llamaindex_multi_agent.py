
import asyncio
import time
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from llama_index.core.agent import ReActAgent
import logging
logger = logging.getLogger(__name__)

import time
import pytest

memory_exporter = InMemorySpanExporter()
span_processors=[SimpleSpanProcessor(memory_exporter)]

@pytest.fixture(scope="module")
def setup():
    memory_exporter.clear()
    setup_monocle_telemetry(
            workflow_name="llamaindex_agent_1", monocle_exporters_list='file',
####            span_processors=[SimpleSpanProcessor(memory_exporter)]
)

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

def setup_agents():
    llm = OpenAI(model="gpt-4")

    flight_tool = FunctionTool.from_defaults(
        fn=book_flight,
        name="book_flight",
        description="Books a flight from one airport to another."
    )
    flight_agent = FunctionAgent(name="flight_booking_agent", tools=[flight_tool], llm=llm,
                            system_prompt="You are a flight booking assistant who books flights based on user requests. You get the flight booking task from the supervisor agent. Once you complete the task, you handoff the response to the supervisor agent.",
                            description="Flight booking agent",
                            can_handoff_to=["supervisor"])

    hotel_tool = FunctionTool.from_defaults(
        fn=book_hotel,
        name="book_hotel",
        description="Books a hotel stay."
    )
    hotel_agent = FunctionAgent(name="hotel_booking_agent", tools=[hotel_tool], llm=llm,
                            system_prompt="You are a hotel booking assistant who books hotels based on user requests. You get the hotel booking task from the supervisor agent. Once you complete the task, you handoff the response to the supervisor agent.",
                            description="Hotel booking agent",
                            can_handoff_to=["supervisor"])

    supervisor = FunctionAgent(name="supervisor", tools=[], llm=llm,
                            system_prompt="You are a supervisor agent who manages the flight and hotel booking agents. You assign tasks to them based on user requests. Once they complete their tasks, you collect their responses and provide a final response to the user.",
                            description="Travel booking supervisor agent",
                            can_handoff_to=["flight_booking_agent", "hotel_booking_agent"])

    agent_workflow = AgentWorkflow(
        agents=[flight_agent, hotel_agent, supervisor],
        root_agent=supervisor.name
    )
    return agent_workflow

async def run_agent():
    """Test multi-agent interaction with flight and hotel booking."""

    agent_workflow = setup_agents()
    resp = await agent_workflow.run(user_msg="book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel")
    print(resp)

@pytest.mark.integration()
def test_multi_agent(setup):
    """Test multi-agent interaction with flight and hotel booking."""

    asyncio.run(run_agent())
    verify_spans()

def verify_spans():
    time.sleep(2)
    found_inference = found_agent = found_tool = False
    found_flight_agent = found_hotel_agent = found_supervisor_agent = False
    found_book_hotel_tool = found_book_flight_tool = False
    found_book_flight_delegation = found_book_hotel_delegation = False
    spans = memory_exporter.get_finished_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4o"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4o"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
            found_inference = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.invocation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.langgraph"
            if span_attributes["entity.1.name"] == "flight_assistant":
                found_flight_agent = True
            elif span_attributes["entity.1.name"] == "hotel_assistant":
                found_hotel_agent = True
            elif span_attributes["entity.1.name"] == "supervisor":
                found_supervisor_agent = True
            found_agent = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.tool":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.langgraph"
            if span_attributes["entity.1.name"] == "book_flight":
                found_book_flight_tool = True
            elif span_attributes["entity.1.name"] == "book_hotel":
                found_book_hotel_tool = True
            found_tool = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.delegation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.langgraph"
            if span_attributes["entity.1.name"] == "flight_assistant":
                found_book_flight_delegation = True
            elif span_attributes["entity.1.name"] == "hotel_assistant":
                found_book_hotel_delegation = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    assert found_book_flight_delegation, "Book flight delegation span not found"
    assert found_book_hotel_delegation, "Book hotel delegation span not found"
    assert found_flight_agent, "Flight assistant agent span not found"
    assert found_hotel_agent, "Hotel assistant agent span not found"
    assert found_supervisor_agent, "Supervisor agent span not found"
    assert found_book_flight_tool, "Book flight tool span not found"
    assert found_book_hotel_tool, "Book hotel tool span not found"

