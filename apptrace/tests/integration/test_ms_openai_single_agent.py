import logging
import os
import random
import time
from typing import Annotated

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

try:
    from agent_framework.openai import OpenAIAssistantsClient

    MICROSOFT_OPENAI_AGENT_AVAILABLE = True
except ImportError:
    MICROSOFT_OPENAI_AGENT_AVAILABLE = False


logger = logging.getLogger(__name__)


def book_flight(
    from_airport: Annotated[str, "The departure airport code (e.g., JFK, LAX)"],
    to_airport: Annotated[str, "The destination airport code (e.g., SFO, ORD)"],
    travel_date: Annotated[str, "Travel date in YYYY-MM-DD format"],
) -> str:
    confirmation = f"FL{random.randint(100000, 999999)}"
    cost = random.randint(300, 900)
    return (
        f"FLIGHT BOOKING CONFIRMED #{confirmation}: {from_airport} to {to_airport} "
        f"on {travel_date} - ${cost}"
    )


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") if MICROSOFT_OPENAI_AGENT_AVAILABLE else None
OPENAI_CHAT_MODEL_ID = (
    os.getenv("OPENAI_CHAT_MODEL_ID")
    or os.getenv("OPENAI_MODEL")
    or "gpt-4o-mini"
)


@pytest.fixture(scope="module")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter),
    ]
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="ms_openai_assistants_test",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.mark.skipif(
    not MICROSOFT_OPENAI_AGENT_AVAILABLE,
    reason="Microsoft Agent Framework OpenAI client not installed",
)
@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_openai_assistants_multi_turn_session(setup):
    client = OpenAIAssistantsClient(
        api_key=OPENAI_API_KEY,
        model_id=OPENAI_CHAT_MODEL_ID,
    )

    flight_agent = client.create_agent(
        name="MS_OpenAI_Flight_Booking_Agent",
        instructions=(
            "You are a Flight Booking Assistant. "
            "Your goal is to help users book flights between any two cities or airports. "
            "Book requested flights and provide confirmation details."
        ),
        tools=[book_flight],
    )

    logger.info("Creating new OpenAI-managed thread...")
    thread = flight_agent.get_new_thread()

    logger.info("First interaction: Book a flight from BOM to JFK on 2026-12-15")
    response1 = await flight_agent.run(
        "Book a flight from BOM to JFK on 2026-12-15",
        thread=thread,
    )
    logger.info(f"Agent Response 1: {response1.text}")
    assert response1 and response1.text, "Should get text response for first interaction"

    service_thread_id = thread.service_thread_id
    logger.info(f"OpenAI Thread ID: {service_thread_id}")
    assert service_thread_id, "Thread should have a service thread ID"
    assert str(service_thread_id).startswith("thread_"), "Thread ID should start with 'thread_'"

    logger.info("Second interaction: Also add a return flight on 2026-12-21")
    response2 = await flight_agent.run(
        "Also add a return flight on 2026-12-21",
        thread=thread,
    )
    logger.info(f"Agent Response 2: {response2.text}")
    assert response2 and response2.text, "Should get text response for second interaction"

    logger.info("Resuming existing thread using service_thread_id")
    resumed_thread = flight_agent.get_new_thread(service_thread_id=service_thread_id)
    assert resumed_thread.service_thread_id == service_thread_id

    logger.info("Third interaction: What did we plan so far?")
    response3 = await flight_agent.run("What did we plan so far?", thread=resumed_thread)
    logger.info(f"Agent Response 3: {response3.text}")
    assert response3 and response3.text, "Should get response from resumed session"

    response_lower = response3.text.lower()
    assert any(
        keyword in response_lower
        for keyword in ["flight", "bom", "jfk", "return", "2026"]
    ), "Resumed conversation should reference previous flight booking context"

    verify_spans_with_session(setup, str(service_thread_id))


def verify_spans_with_session(custom_exporter, expected_thread_id: str):
    time.sleep(2)

    found_inference = False
    found_agent = False
    found_tool = False
    found_session_attribute = False
    session_ids_found = set()

    spans = custom_exporter.get_captured_spans()
    logger.info(f"Analyzing {len(spans)} spans for OpenAI session tracking...")

    for span in spans:
        span_attributes = span.attributes

        if "scope.agentic.session" in span_attributes:
            found_session_attribute = True
            session_id = str(span_attributes["scope.agentic.session"])
            session_ids_found.add(session_id)
            logger.info(f"Found session ID in span '{span.name}': {session_id}")

        if "span.type" in span_attributes and span_attributes["span.type"] in [
            "inference",
            "inference.framework",
            "inference.modelapi",
        ]:
            found_inference = True

        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.invocation"
            and "entity.1.name" in span_attributes
        ):
            assert span_attributes.get("entity.1.type") == "agent.microsoft", (
                "Agent entity should be of type agent.microsoft"
            )
            assert span_attributes["entity.1.name"] == "MS_OpenAI_Flight_Booking_Agent", (
                "Agent name should be MS_OpenAI_Flight_Booking_Agent"
            )
            found_agent = True

        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            assert span_attributes.get("entity.1.type") == "tool.microsoft", (
                "Tool entity should be of type tool.microsoft"
            )
            assert span_attributes["entity.1.name"] == "book_flight", "Tool name should be book_flight"
            found_tool = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent invocation span not found"
    assert found_tool, "Tool invocation span not found"
    assert found_session_attribute, "scope.agentic.session attribute NOT FOUND in any spans"
    assert expected_thread_id in session_ids_found, (
        f"Expected thread ID '{expected_thread_id}' not found in session IDs: {session_ids_found}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
