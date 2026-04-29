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
    def book_hotel_tool(hotel_name: str) -> str:
        return book_hotel(hotel_name)

    hotel_agent = Agent(
        name="Hotel Agent",
        instructions="You are a hotel booking specialist. Use the hotel tool to book hotels.",
        tools=[book_hotel_tool],
        model_settings=ModelSettings(tool_choice="required"),
    )

    streamed_result = Runner.run_streamed(
        hotel_agent,
        "Book a hotel in Los Angeles for 2 nights.",
    )
    if asyncio.iscoroutine(streamed_result):
        streamed_result = await streamed_result

    stream_events = 0
    async for _ in streamed_result.stream_events():
        stream_events += 1

    final_output = getattr(streamed_result, "final_output", None)
    if not final_output and hasattr(streamed_result, "get_final_output"):
        maybe_output = streamed_result.get_final_output()
        final_output = await maybe_output if asyncio.iscoroutine(maybe_output) else maybe_output

    logger.info("Observed %s streamed events", stream_events)
    assert stream_events > 0, "Expected at least one streamed event"
    assert final_output is not None, "Expected final streamed output"

    verify_streaming_agent_spans(setup)


def verify_streaming_agent_spans(memory_exporter):
    time.sleep(2)
    spans = memory_exporter.get_finished_spans()

    inference_spans = [s for s in spans if s.attributes.get("span.type") in {"inference", "inference.framework", "inference.modelapi"}]
    agent_spans = [s for s in spans if s.attributes.get("span.type") == "agentic.invocation"]
    tool_spans = [s for s in spans if s.attributes.get("span.type") == "agentic.tool.invocation"]

    assert inference_spans, "Inference span not found"
    assert agent_spans, "Agent invocation span not found"
    assert tool_spans, "Tool span not found"

    assert any(s.attributes.get("entity.1.name") == "Hotel Agent" for s in agent_spans), "Hotel Agent invocation span not found"
    assert any(s.attributes.get("entity.1.name") == "book_hotel_tool" for s in tool_spans), "Hotel tool span not found"
    assert any(s.attributes.get("entity.2.type", "").startswith("model.llm.") for s in inference_spans), "Model entity missing in inference span"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
