import asyncio
import datetime
import logging
import os
import time
from zoneinfo import ZoneInfo

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from monocle_apptrace import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup():
    try:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        memory_exporter = InMemorySpanExporter()
        custom_exporter = CustomConsoleSpanExporter()
        span_processors = [SimpleSpanProcessor(memory_exporter), SimpleSpanProcessor(custom_exporter)]
        instrumentor = setup_monocle_telemetry(
            workflow_name="langchain_agent_1",
            span_processors=span_processors
        )
        yield memory_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def book_flight(from_airport: str, to_airport: str) -> dict:
    """Books a flight from one airport to another.

    Args:
        from_airport (str): The airport from which the flight departs.
        to_airport (str): The airport to which the flight arrives.

    Returns:
        dict: status and message.
    """
    return {
        "status": "success",
        "message": f"Flight booked from {from_airport} to {to_airport}."
    }

def book_hotel(hotel_name: str, city: str) -> dict:
    """Books a hotel for a stay.

    Args:
        hotel_name (str): The name of the hotel to book.
        city (str): The city where the hotel is located.

    Returns:
        dict: status and message.
    """
    return {
        "status": "success",
        "message": f"Successfully booked a stay at {hotel_name} in {city}."
    }

flight_booking_agent = LlmAgent(
    name="flight_assistant",
    model="gemini-2.0-flash",
    description=(
        "Agent to book flights based on user queries."
    ),
    instruction=(
        "You are a helpful agent who can assist users in booking flights."
    ),
    tools=[book_flight]  # Define flight booking tools here
)

hotel_booking_agent = LlmAgent(
    name="hotel_assistant",
    model="gemini-2.0-flash",
    description=(
        "Agent to book hotels based on user queries."
    ),
    instruction=(
        "You are a helpful agent who can assist users in booking hotels."
    ),
    tools=[book_hotel]  # Define hotel booking tools here
)

root_agent = SequentialAgent(
    name="supervisor",
    description=(
        "Supervisor agent that coordinates the flight booking and hotel booking. Provide a consolidated response."
    ),
    sub_agents=[flight_booking_agent, hotel_booking_agent],
)

session_service = InMemorySessionService()
APP_NAME = "streaming_app"
USER_ID = "user_123"
SESSION_ID = "session_456"

runner = Runner(
    agent=root_agent,  # Assume this is defined
    app_name=APP_NAME,
    session_service=session_service
)

async def run_agent(test_message: str):
    session = await session_service.create_session(
        app_name=APP_NAME, 
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    content = types.Content(role='user', parts=[types.Part(text=test_message)])
    # Process events as they arrive using async for
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        # For final response
        if event.is_final_response():
            logger.info(event.content)  # End line after response

@pytest.mark.asyncio
async def test_multi_agent(setup):
    test_message = "Book a flight from San Francisco to Mumbai, book Taj Mahal hotel in Mumbai."
    await run_agent(test_message)
    verify_spans(setup)

def verify_spans(memory_exporter):
    time.sleep(2)
    found_inference = found_agent = found_tool = found_delegation = False
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
            assert span_attributes["entity.1.type"] == "inference.gemini"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gemini-2.0-flash"
            assert span_attributes["entity.2.type"] == "model.llm.gemini-2.0-flash"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "input" in span_input.attributes
            assert span_input.attributes["input"] is not None and span_input.attributes["input"] != ""
            assert "response" in span_output.attributes
            assert span_output.attributes["response"] is not None and span_output.attributes["response"] != ""
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
            assert "finish_type" in span_metadata.attributes
            assert "finish_reason" in span_metadata.attributes
            found_inference = True

        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.invocation"
                and "entity.1.name" in span_attributes
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.adk"
            if span_attributes["entity.1.name"] == "flight_assistant":
                found_flight_agent = True
                if span_attributes["entity.1.from_agent"] == "supervisor":
                    found_book_flight_delegation = True
            elif span_attributes["entity.1.name"] == "hotel_assistant":
                found_hotel_agent = True
                if span_attributes["entity.1.from_agent"] == "supervisor":
                    found_book_hotel_delegation = True
            elif span_attributes["entity.1.name"] == "supervisor":
                found_supervisor_agent = True
        found_agent = True

        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.adk"
            if span_attributes["entity.1.name"] == "book_flight":
                found_book_flight_tool = True
            elif span_attributes["entity.1.name"] == "book_hotel":
                found_book_hotel_tool = True
            found_tool = True
            span_input, span_output = span.events
            assert "input" in span_input.attributes
            assert "response" in span_output.attributes

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    assert found_flight_agent, "Flight assistant agent span not found"
    assert found_hotel_agent, "Hotel assistant agent span not found"
    assert found_supervisor_agent, "Supervisor agent span not found"
    assert found_book_flight_delegation, "Book flight delegation span not found"
    assert found_book_hotel_delegation, "Book hotel delegation span not found"
    assert found_book_flight_tool, "Book flight tool span not found"
    assert found_book_hotel_tool, "Book hotel tool span not found"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])