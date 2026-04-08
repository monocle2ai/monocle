import asyncio
import logging
import time

import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup():
    memory_exporter = InMemorySpanExporter()
    custom_exporter = CustomConsoleSpanExporter()
    span_processors = [
        SimpleSpanProcessor(memory_exporter),
        BatchSpanProcessor(FileSpanExporter()),
        SimpleSpanProcessor(custom_exporter),
    ]
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="openai_streaming_multi_agent",
            span_processors=span_processors,
        )
        yield memory_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def book_flight(from_airport: str, to_airport: str) -> str:
    return f"Successfully booked a flight from {from_airport} to {to_airport} for 100 USD."


def book_hotel(hotel_name: str) -> str:
    return f"Successfully booked a stay at {hotel_name} for 50 USD."


@pytest.mark.asyncio
async def test_agents_sdk_streaming_multi_agent(setup):
    try:
        from agents import Agent, Runner, ModelSettings, function_tool
    except ImportError:
        pytest.skip("OpenAI Agents SDK not available")

    if not hasattr(Runner, "run_streamed"):
        pytest.skip("Installed OpenAI Agents SDK does not support Runner.run_streamed")

    @function_tool
    def book_flight_tool(from_airport: str, to_airport: str) -> str:
        return book_flight(from_airport, to_airport)

    @function_tool
    def book_hotel_tool(hotel_name: str) -> str:
        return book_hotel(hotel_name)

    flight_agent = Agent(
        name="Flight Agent",
        instructions="You are a flight booking specialist. Use the flight tool to book flights.",
        tools=[book_flight_tool],
        model_settings=ModelSettings(tool_choice="required"),
    )

    hotel_agent = Agent(
        name="Hotel Agent",
        instructions="You are a hotel booking specialist. Use the hotel tool to book hotels.",
        tools=[book_hotel_tool],
        model_settings=ModelSettings(tool_choice="required"),
    )

    coordinator = Agent(
        name="Travel Coordinator",
        instructions=(
            "You are a travel coordinator. Always delegate flight work to Flight Agent and hotel work to Hotel Agent. "
            "For combined requests, invoke both handoffs before producing a final response."
        ),
        handoffs=[flight_agent, hotel_agent],
    )

    async def _run_streamed_prompt(prompt: str):
        streamed_result = Runner.run_streamed(coordinator, prompt)
        if asyncio.iscoroutine(streamed_result):
            streamed_result = await streamed_result

        stream_events = 0
        content_fragments = []

        if hasattr(streamed_result, "stream_events"):
            event_source = streamed_result.stream_events()
        else:
            event_source = streamed_result

        async for event in event_source:
            stream_events += 1
            event_type = getattr(event, "type", "")
            if event_type == "raw_response_event":
                data = getattr(event, "data", None)
                text_delta = getattr(data, "delta", None)
                if text_delta:
                    content_fragments.append(str(text_delta))

        final_output = getattr(streamed_result, "final_output", None)
        if not final_output and hasattr(streamed_result, "get_final_output"):
            maybe_output = streamed_result.get_final_output()
            final_output = await maybe_output if asyncio.iscoroutine(maybe_output) else maybe_output

        logger.info("Observed %s streamed events for prompt: %s", stream_events, prompt)
        assert stream_events > 0, "Expected at least one streamed event"
        assert final_output is not None or len(content_fragments) > 0, "Expected streamed output"

    await _run_streamed_prompt(
        "Book a flight from NYC to LAX and book a hotel in Los Angeles. "
        "You must call both specialists and include both booking confirmations."
    )

    # Explicit second prompt ensures the hotel specialist path is exercised even when routing is conservative.
    await _run_streamed_prompt("Now only book a hotel in Los Angeles using Hotel Agent.")

    verify_streaming_agent_spans(setup)


def verify_streaming_agent_spans(memory_exporter):
    time.sleep(2)
    spans = memory_exporter.get_finished_spans()

    found_inference = False
    found_agent = False
    found_tool = False
    found_hotel_agent = False

    for span in spans:
        span_attributes = span.attributes

        if span_attributes.get("span.type") in {"inference", "inference.framework", "inference.modelapi"}:
            found_inference = True

        if span_attributes.get("span.type") == "agentic.invocation":
            found_agent = True
            if span_attributes.get("entity.1.name") == "Hotel Agent":
                found_hotel_agent = True

        if span_attributes.get("span.type") == "agentic.tool.invocation":
            found_tool = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent invocation span not found"
    assert found_tool, "Tool span not found"
    assert found_hotel_agent, "Hotel Agent invocation span not found"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
