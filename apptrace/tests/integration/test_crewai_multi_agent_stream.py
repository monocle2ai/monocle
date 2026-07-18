import logging
import time
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

# Import CrewAI components
try:
    from crewai import Agent, Crew, Task
    from crewai.tools import BaseTool
    from crewai.types.streaming import CrewStreamingOutput
    from langchain_openai import ChatOpenAI

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class BookHotelTool(BaseTool):
    name: str = "book_hotel"
    description: str = "Book a hotel reservation"

    def _run(self, hotel_name: str, city: str, nights: int = 1) -> str:
        import random
        import time

        time.sleep(0.1)
        confirmation = f"HT{random.randint(100000, 999999)}"
        cost = nights * 150
        return f"HOTEL BOOKING CONFIRMED #{confirmation}: {hotel_name} in {city} for {nights} nights - ${cost}"


class BookFlightTool(BaseTool):
    name: str = "book_flight"
    description: str = "Book a flight reservation"

    def _run(self, departure: str, arrival: str, date: str = "next week") -> str:
        import random
        import time

        time.sleep(0.1)
        confirmation = f"FL{random.randint(100000, 999999)}"
        cost = random.randint(300, 800)
        return f"FLIGHT BOOKING CONFIRMED #{confirmation}: {departure} to {arrival} on {date} - ${cost}"


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry instrumentation for CrewAI multi-agent streaming tests."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter),
    ]

    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="crewai_multi_agent_stream_test",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def create_multi_agent_stream_crew():
    """Create a multi-agent CrewAI crew configured for streaming output."""
    if not CREWAI_AVAILABLE:
        pytest.skip("CrewAI not available")

    hotel_tool = BookHotelTool()
    flight_tool = BookFlightTool()

    hotel_agent = Agent(
        role="Hotel Booking Agent",
        goal="Book the best hotel accommodations for travelers",
        backstory="You are an expert hotel booking agent with extensive knowledge of hotels worldwide.",
        tools=[hotel_tool],
        llm=ChatOpenAI(model="gpt-4.1"),
        verbose=True,
        allow_delegation=False,
    )

    flight_agent = Agent(
        role="Flight Booking Agent",
        goal="Book the best flight options for travelers",
        backstory="You are an expert flight booking agent with access to all major airlines.",
        tools=[flight_tool],
        llm=ChatOpenAI(model="gpt-4.1"),
        verbose=True,
        allow_delegation=False,
    )

    supervisor_agent = Agent(
        role="Travel Supervisor",
        goal="Coordinate complete travel bookings by delegating to agents",
        backstory="You are a travel supervisor who manages a team of booking agents. You delegate hotel bookings to the hotel agent and flight bookings to the flight agent.",
        llm=ChatOpenAI(model="gpt-4.1"),
        verbose=True,
        allow_delegation=False,
    )

    hotel_task = Task(
        name="Hotel Booking Task",
        description="Book a hotel room at Marriott in New York for 2 nights.",
        expected_output="Hotel booking confirmation with confirmation number and cost",
        agent=hotel_agent,
    )

    flight_task = Task(
        name="Flight Booking Task",
        description="Book a flight from Los Angeles to New York for next week.",
        expected_output="Flight booking confirmation with confirmation number and cost",
        agent=flight_agent,
    )

    supervisor_task = Task(
        name="Travel Booking Supervision Task",
        description="Coordinate a complete travel package: book both flight from Los Angeles to New York and hotel accommodation at Marriott in New York for 2 nights. Ensure both bookings are completed successfully.",
        expected_output="Complete travel booking summary with both flight and hotel confirmations",
        agent=supervisor_agent,
    )

    crew = Crew(
        agents=[supervisor_agent, hotel_agent, flight_agent],
        tasks=[hotel_task, flight_task, supervisor_task],
        verbose=True,
        process="sequential",
        stream=True,
    )
    return crew


def _has_booking_signals(result_text: str) -> bool:
    has_hotel = "HOTEL BOOKING CONFIRMED" in result_text or "hotel" in result_text.lower()
    has_flight = "FLIGHT BOOKING CONFIRMED" in result_text or "flight" in result_text.lower()
    return has_hotel and has_flight


@pytest.mark.skipif(not CREWAI_AVAILABLE, reason="CrewAI not installed")
def test_crewai_multi_agent_stream(setup):
    """Sync streaming CrewAI test with 2 agents, 2 tools, and 1 supervisor."""
    crew = create_multi_agent_stream_crew()

    stream_output = crew.kickoff(
        inputs={
            "travel_request": "Plan a complete trip from Los Angeles to New York with hotel stay"
        }
    )
    assert isinstance(stream_output, CrewStreamingOutput), "Expected a streaming output object"

    streamed_chunks = []
    for chunk in stream_output:
        chunk_text = getattr(chunk, "content", "")
        if chunk_text:
            streamed_chunks.append(chunk_text)

    assert stream_output.is_completed, "Stream should be completed after iteration"
    assert len(streamed_chunks) > 0, "Should receive at least one streamed chunk"

    result = stream_output.result
    assert result is not None, "Should get a final result"
    result_str = str(result)
    assert _has_booking_signals(result_str), "Final result should include both hotel and flight booking details"

    logger.info(f"Streamed {len(streamed_chunks)} chunks")
    logger.info(f"Streaming final result: {result}")
    verify_stream_spans(setup)


@pytest.mark.skipif(not CREWAI_AVAILABLE, reason="CrewAI not installed")
@pytest.mark.asyncio
async def test_crewai_multi_agent_stream_async(setup):
    """Async streaming CrewAI test."""
    crew = create_multi_agent_stream_crew()

    stream_output = await crew.kickoff_async(
        inputs={
            "travel_request": "Plan a complete trip from Los Angeles to New York with hotel stay"
        }
    )
    assert isinstance(stream_output, CrewStreamingOutput), "Expected a streaming output object"

    streamed_chunks = []
    async for chunk in stream_output:
        chunk_text = getattr(chunk, "content", "")
        if chunk_text:
            streamed_chunks.append(chunk_text)

    assert stream_output.is_completed, "Async stream should be completed after iteration"
    assert len(streamed_chunks) > 0, "Should receive at least one async streamed chunk"

    result = stream_output.result
    assert result is not None, "Should get a final async result"
    result_str = str(result)
    assert _has_booking_signals(result_str), "Final async result should include both hotel and flight booking details"

    logger.info(f"Async streamed {len(streamed_chunks)} chunks")
    logger.info(f"Async streaming final result: {result}")
    verify_stream_spans(setup)


def verify_stream_spans(custom_exporter):
    """Verify that task, tool, and inference spans are emitted in streaming mode."""
    time.sleep(2)
    found_inference = found_agent = found_tool = False
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        attrs = span.attributes
        span_type = attrs.get("span.type", "")

        if span_type in ("inference", "inference.framework"):
            assert attrs.get("entity.1.type") == "inference.openai"
            assert "entity.1.provider_name" in attrs
            assert "entity.1.inference_endpoint" in attrs
            assert "entity.2.name" in attrs
            assert "entity.2.type" in attrs
            found_inference = True

        if span_type == "agentic.invocation" and "entity.1.name" in attrs:
            assert attrs.get("entity.1.type") == "agent.crewai"
            # Both Agent.execute_task spans (agent role names) and
            # Task.execute_sync spans (task names) emit agentic.invocation
            known_names = (
                "Hotel Booking Agent", "Flight Booking Agent", "Travel Supervisor",
                "Hotel Booking Task", "Flight Booking Task", "Travel Booking Supervision Task",
            )
            assert attrs["entity.1.name"] in known_names, (
                f"Unexpected agentic.invocation name: {attrs['entity.1.name']}"
            )
            found_agent = True

        if span_type == "agentic.tool.invocation":
            assert attrs.get("entity.1.type") == "tool.crewai"
            assert attrs.get("entity.1.name") in ("book_hotel", "book_flight")
            found_tool = True

    assert found_inference, "No inference span captured in streaming mode"
    assert found_agent, "No agentic.invocation span captured in streaming mode"
    assert found_tool, "No agentic.tool.invocation span captured in streaming mode"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

