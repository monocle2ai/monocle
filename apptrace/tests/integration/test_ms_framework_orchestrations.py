"""Integration tests for Microsoft Agent Framework orchestration builders.

Covers the agent-framework-orchestrations package (SequentialBuilder, ConcurrentBuilder,
GroupChatBuilder, MagenticBuilder, HandoffBuilder). These builders reuse the core Workflow
and AgentExecutor primitives (already instrumented) and add coordinator/manager executors
(GroupChatOrchestrator / AgentBasedGroupChatOrchestrator / MagenticOrchestrator) that Monocle
traces as agentic.invocation spans with the "routing" subtype.

Tests are skipped unless the orchestration package is installed and Azure OpenAI credentials
are configured.
"""

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
    from agent_framework.openai import OpenAIChatCompletionClient
    from agent_framework.orchestrations import (
        ConcurrentBuilder,
        GroupChatBuilder,
        HandoffBuilder,
        MagenticBuilder,
        SequentialBuilder,
    )

    MICROSOFT_ORCHESTRATIONS_AVAILABLE = True
except ImportError:
    MICROSOFT_ORCHESTRATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


def book_flight(
    from_airport: Annotated[str, "The departure airport code (e.g., JFK, LAX)"],
    to_airport: Annotated[str, "The destination airport code (e.g., SFO, ORD)"],
) -> str:
    """Book a flight from one airport to another"""
    confirmation = f"FL{random.randint(100000, 999999)}"
    cost = random.randint(300, 800)
    return f"FLIGHT BOOKING CONFIRMED #{confirmation}: {from_airport} to {to_airport} - ${cost}"


def book_hotel(
    hotel_name: Annotated[str, "The name of the hotel to book"],
    city: Annotated[str, "The city where the hotel is located"],
    nights: Annotated[int, "Number of nights to stay"] = 1,
) -> str:
    """Book a hotel reservation"""
    confirmation = f"HT{random.randint(100000, 999999)}"
    cost = nights * 150
    return f"HOTEL BOOKING CONFIRMED #{confirmation}: {hotel_name} in {city} for {nights} nights - ${cost}"


azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") if MICROSOFT_ORCHESTRATIONS_AVAILABLE else None
model = os.getenv("AZURE_OPENAI_API_DEPLOYMENT") if MICROSOFT_ORCHESTRATIONS_AVAILABLE else None
api_key = os.getenv("AZURE_OPENAI_API_KEY") if MICROSOFT_ORCHESTRATIONS_AVAILABLE else None
api_version = (
    os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if MICROSOFT_ORCHESTRATIONS_AVAILABLE
    else None
)

CREDENTIALS_AVAILABLE = bool(MICROSOFT_ORCHESTRATIONS_AVAILABLE and azure_endpoint and model)

if CREDENTIALS_AVAILABLE:
    client = OpenAIChatCompletionClient(
        model=model,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    flight_agent = client.as_agent(
        name="MS_Flight_Booking_Agent",
        instructions=(
            "You are a Flight Booking Assistant. Book the requested flight and "
            "provide confirmation details."
        ),
        tools=[book_flight],
    )
    hotel_agent = client.as_agent(
        name="MS_Hotel_Booking_Agent",
        instructions=(
            "You are a Hotel Booking Assistant. Book the requested hotel and "
            "provide confirmation details."
        ),
        tools=[book_hotel],
    )
    summarizer_agent = client.as_agent(
        name="MS_Travel_Summarizer",
        instructions=(
            "You are a Travel Booking Summarizer. Review all booking confirmations "
            "and create a consolidated summary with confirmation numbers and total costs."
        ),
        tools=[],
    )
    manager_agent = client.as_agent(
        name="MS_Travel_Manager",
        instructions=(
            "You coordinate a team of travel booking agents to complete travel "
            "arrangements efficiently."
        ),
        tools=[],
    )
else:
    client = None
    flight_agent = hotel_agent = summarizer_agent = manager_agent = None


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry instrumentation for Microsoft Agent Framework orchestration tests."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter),
    ]
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="microsoft_agent_orchestrations_test",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


TASK = "Book a flight from BOM to JFK and a stay at the Marriott for 3 nights."


def _agent_names(spans):
    """Return the set of agent-invocation span names captured."""
    names = set()
    for span in spans:
        attrs = span.attributes
        if attrs.get("span.type") == "agentic.invocation" and "entity.1.name" in attrs:
            names.add(attrs["entity.1.name"])
    return names


def _has_orchestrator_span(spans, pattern=None):
    """Return True when a coordinator/manager routing span was captured."""
    for span in spans:
        attrs = span.attributes
        if (
            attrs.get("span.type") == "agentic.invocation"
            and attrs.get("span.subtype") == "routing"
            and attrs.get("entity.1.type") == "agent.microsoft"
        ):
            if pattern is None or attrs.get("entity.1.description") == pattern:
                return True
    return False


def _assert_common(custom_exporter):
    """Assert the baseline spans every orchestration should emit."""
    time.sleep(2)
    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans captured"

    turn_count = sum(1 for s in spans if s.attributes.get("span.type") == "agentic.turn")
    assert turn_count >= 1, "Expected at least one agentic.turn span"
    return spans


@pytest.mark.skipif(
    not MICROSOFT_ORCHESTRATIONS_AVAILABLE,
    reason="agent-framework-orchestrations not installed",
)
@pytest.mark.asyncio
async def test_sequential_orchestration(setup):
    """SequentialBuilder chains agents; participants surface as agent invocations."""
    if not CREDENTIALS_AVAILABLE:
        pytest.skip("Azure OpenAI credentials not configured")

    workflow = SequentialBuilder(participants=[flight_agent, hotel_agent, summarizer_agent]).build()
    response = await workflow.run(TASK)
    assert response, "Should get a workflow response"

    spans = _assert_common(setup)
    names = _agent_names(spans)
    assert "MS_Flight_Booking_Agent" in names, "Flight agent invocation span not found"
    assert "MS_Hotel_Booking_Agent" in names, "Hotel agent invocation span not found"


@pytest.mark.skipif(
    not MICROSOFT_ORCHESTRATIONS_AVAILABLE,
    reason="agent-framework-orchestrations not installed",
)
@pytest.mark.asyncio
async def test_concurrent_orchestration(setup):
    """ConcurrentBuilder fans out to agents in parallel; each surfaces as an invocation."""
    if not CREDENTIALS_AVAILABLE:
        pytest.skip("Azure OpenAI credentials not configured")

    workflow = ConcurrentBuilder(participants=[flight_agent, hotel_agent]).build()
    response = await workflow.run(TASK)
    assert response, "Should get a workflow response"

    spans = _assert_common(setup)
    names = _agent_names(spans)
    assert "MS_Flight_Booking_Agent" in names, "Flight agent invocation span not found"
    assert "MS_Hotel_Booking_Agent" in names, "Hotel agent invocation span not found"


@pytest.mark.skipif(
    not MICROSOFT_ORCHESTRATIONS_AVAILABLE,
    reason="agent-framework-orchestrations not installed",
)
@pytest.mark.asyncio
async def test_group_chat_orchestration(setup):
    """GroupChatBuilder with an orchestrator agent emits a coordinator routing span."""
    if not CREDENTIALS_AVAILABLE:
        pytest.skip("Azure OpenAI credentials not configured")

    workflow = GroupChatBuilder(
        participants=[flight_agent, hotel_agent],
        orchestrator_agent=manager_agent,
        max_rounds=6,
    ).build()
    response = await workflow.run(TASK)
    assert response, "Should get a workflow response"

    spans = _assert_common(setup)
    assert _has_orchestrator_span(spans, pattern="group_chat"), "Group chat coordinator span not found"


@pytest.mark.skipif(
    not MICROSOFT_ORCHESTRATIONS_AVAILABLE,
    reason="agent-framework-orchestrations not installed",
)
@pytest.mark.asyncio
async def test_magentic_orchestration(setup):
    """MagenticBuilder emits a MagenticOrchestrator routing span alongside participants."""
    if not CREDENTIALS_AVAILABLE:
        pytest.skip("Azure OpenAI credentials not configured")

    workflow = MagenticBuilder(
        participants=[flight_agent, hotel_agent],
        manager_agent=manager_agent,
        max_round_count=6,
        max_stall_count=2,
    ).build()
    response = await workflow.run(TASK)
    assert response, "Should get a workflow response"

    spans = _assert_common(setup)
    assert _has_orchestrator_span(spans, pattern="magentic"), "Magentic coordinator span not found"


@pytest.mark.skipif(
    not MICROSOFT_ORCHESTRATIONS_AVAILABLE,
    reason="agent-framework-orchestrations not installed",
)
@pytest.mark.asyncio
async def test_handoff_orchestration(setup):
    """HandoffBuilder routes between agents; participants surface as agent invocations."""
    if not CREDENTIALS_AVAILABLE:
        pytest.skip("Azure OpenAI credentials not configured")

    # Handoff requires per-service-call history persistence on every participant.
    handoff_flight_agent = client.as_agent(
        name="MS_Flight_Booking_Agent",
        instructions=(
            "You are a Flight Booking Assistant. Book the requested flight and "
            "provide confirmation details."
        ),
        tools=[book_flight],
        require_per_service_call_history_persistence=True,
    )
    handoff_hotel_agent = client.as_agent(
        name="MS_Hotel_Booking_Agent",
        instructions=(
            "You are a Hotel Booking Assistant. Book the requested hotel and "
            "provide confirmation details."
        ),
        tools=[book_hotel],
        require_per_service_call_history_persistence=True,
    )

    workflow = (
        HandoffBuilder(participants=[handoff_flight_agent, handoff_hotel_agent])
        .with_start_agent(handoff_flight_agent)
        .build()
    )
    response = await workflow.run(TASK)
    assert response, "Should get a workflow response"

    spans = _assert_common(setup)
    names = _agent_names(spans)
    assert "MS_Flight_Booking_Agent" in names, "Flight agent invocation span not found"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
