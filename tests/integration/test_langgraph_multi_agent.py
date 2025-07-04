

import asyncio
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
import pytest
import os
import logging
logger = logging.getLogger(__name__)

import time
memory_exporter = InMemorySpanExporter()
span_processors=[SimpleSpanProcessor(memory_exporter)]

@pytest.fixture(scope="module")
def setup():
    memory_exporter.clear()
    setup_monocle_telemetry(
            workflow_name="langchain_agent_1",
            span_processors=[SimpleSpanProcessor(memory_exporter)]
)

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

def setup_agents():

    flight_assistant = create_react_agent(
        model="openai:gpt-4o",
        tools=[book_flight],
        prompt="You are a flight booking assistant",
        name="flight_assistant"
    )

    hotel_assistant = create_react_agent(
        model="openai:gpt-4o",
        tools=[book_hotel],
        prompt="You are a hotel booking assistant",
        name="hotel_assistant"
    )

    supervisor = create_supervisor(
        agents=[flight_assistant, hotel_assistant],
        model=ChatOpenAI(model="gpt-4o"),
        prompt=(
            "You manage a hotel booking assistant and a"
            "flight booking assistant. Assign work to them."
        )
    ).compile()

    return supervisor

@pytest.mark.integration()
def test_multi_agent(setup):
    """Test multi-agent interaction with flight and hotel booking."""

    supervisor = setup_agents()
    chunk = supervisor.invoke(
        input ={
            "messages": [
                {
                    "role": "user",
                    "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
                }
            ]
        }
    )
    print(chunk)
    print("\n")
    verify_spans()

@pytest.mark.integration()
def test_async_multi_agent(setup):
    """Test multi-agent interaction with flight and hotel booking."""

    supervisor = setup_agents()
    chunk = asyncio.run(supervisor.ainvoke(
        input ={
            "messages": [
                {
                    "role": "user",
                    "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
                }
            ]
        }
    ))
    print(chunk)
    print("\n")
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

