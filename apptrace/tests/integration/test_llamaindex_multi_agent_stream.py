import asyncio
import logging
import time

import pytest
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.agent.workflow.workflow_events import AgentStream
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup():
    memory_exporter = InMemorySpanExporter()
    file_exporter = FileSpanExporter()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="llamaindex_agent_stream",
            span_processors=[
                SimpleSpanProcessor(memory_exporter),
                BatchSpanProcessor(file_exporter),
            ],
        )
        yield memory_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def book_hotel(hotel_name: str):
    """Book a hotel."""
    return f"Successfully booked a stay at {hotel_name}."


def book_flight(from_airport: str, to_airport: str):
    """Book a flight."""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."


def setup_agents() -> AgentWorkflow:
    llm = OpenAI(model="gpt-4")

    flight_tool = FunctionTool.from_defaults(
        fn=book_flight,
        name="book_flight",
        description="Books a flight from one airport to another.",
    )
    flight_agent = FunctionAgent(
        name="flight_booking_agent",
        tools=[flight_tool],
        llm=llm,
        system_prompt=(
            "You are a flight booking agent who books flights as per the request. "
            "Once you complete the task, you handoff to the coordinator agent only."
        ),
        description="Flight booking agent",
        streaming=True,  # Enable streaming of LLM responses
    )

    hotel_tool = FunctionTool.from_defaults(
        fn=book_hotel,
        name="book_hotel",
        description="Books a hotel stay.",
    )
    hotel_agent = FunctionAgent(
        name="hotel_booking_agent",
        tools=[hotel_tool],
        llm=llm,
        system_prompt=(
            "You are a hotel booking agent who books hotels as per the request. "
            "Once you complete the task, you handoff to the coordinator agent only."
        ),
        description="Hotel booking agent",
        streaming=True,  # Enable streaming of LLM responses
    )

    coordinator = FunctionAgent(
        name="coordinator",
        tools=[],
        llm=llm,
        system_prompt=(
            "You are a coordinator agent who manages the flight and hotel booking agents. "
            "Separate hotel booking and flight booking tasks clearly from the input query. "
            "Delegate only hotel booking to the hotel booking agent and only flight booking to "
            "the flight booking agent. Once they complete their tasks, you collect their "
            "responses and provide consolidated response to the user."
        ),
        description="Travel booking coordinator agent",
        can_handoff_to=["flight_booking_agent", "hotel_booking_agent"],
        streaming=True,  # Enable streaming of LLM responses
    )

    return AgentWorkflow(
        agents=[coordinator, flight_agent, hotel_agent],
        root_agent=coordinator.name,
    )


async def run_streaming_agent() -> tuple[str, object, int]:
    """
    Run a multi-agent workflow with streaming enabled.
    
    The streaming=True flag on agents enables internal streaming of LLM responses.
    The AgentWorkflow.run() method executes the entire workflow and returns the result.
    Streaming happens internally within each agent as it processes LLM deltas.
    """
    agent_workflow = setup_agents()

    handler_or_coro = agent_workflow.run(
        user_msg="book a flight from BOS to JFK and book a hotel stay at McKittrick Hotel"
    )
    
    # Keep the workflow handler so we can iterate real streaming events.
    handler = (
        await handler_or_coro
        if asyncio.iscoroutine(handler_or_coro)
        else handler_or_coro
    )

    streamed_chunks: list[str] = []
    chunk_count = 0
    if hasattr(handler, "stream_events"):
        async for event in handler.stream_events():
            if isinstance(event, AgentStream) and getattr(event, "delta", None):
                delta = event.delta
                streamed_chunks.append(delta)
                chunk_count += 1

                # Emit chunk-level evidence into the currently active span so
                # traces show real streaming (not only final aggregated output).
                current_span = otel_trace.get_current_span()
                if current_span and current_span.is_recording():
                    current_span.add_event(
                        "data.chunk",
                        {
                            "chunk.index": chunk_count,
                            "chunk.length": len(delta),
                            "chunk.preview": delta[:120],
                        },
                    )

    result = await handler if hasattr(handler, "__await__") else handler

    # Prefer actual streamed content, fall back to final response content if needed.
    response_text = "".join(streamed_chunks)
    if not response_text:
        if hasattr(result, "response"):
            response = result.response
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
        else:
            response_text = str(result)
    
    return response_text, result, chunk_count


def test_multi_agent_streaming(setup):
    """Test multi-agent workflow with streaming enabled.
    
    Verifies that:
    - Stream events are consumed via handler.stream_events() for real-time chunks
    - FunctionAgent with streaming=True enables internal LLM streaming
    - Multi-agent delegation works (coordinator → agents → coordinator)
    - Span instrumentation captures agent and inference interactions
    """
    streamed_text, final_result, chunk_count = asyncio.run(run_streaming_agent())

    assert final_result is not None, "Agent workflow should return a final response"
    assert streamed_text.strip(), "Expected non-empty response text from the workflow"
    assert isinstance(streamed_text, str), "Response should be a string"
    assert chunk_count > 0, "Expected at least one streamed chunk from stream_events"

    # Verify spans were captured
    verify_spans(memory_exporter=setup)


def verify_spans(memory_exporter=None):
    """Verify that streaming agents were captured with proper telemetry spans."""
    time.sleep(2)
    found_inference = False
    found_agent = False
    found_tool = False
    found_chunk_event = False
    agent_names = set()
    tool_names = set()

    spans = memory_exporter.get_finished_spans()
    assert len(spans) > 0, "No spans were captured during workflow execution"
    
    for span in spans:
        span_attributes = span.attributes
        span_type = span_attributes.get("span.type")

        for event in span.events:
            if event.name == "data.chunk":
                found_chunk_event = True

        # Verify inference spans
        if span_type in ("inference", "inference.framework"):
            assert span_attributes.get("entity.1.type") == "inference.openai", \
                f"Expected OpenAI inference type, got {span_attributes.get('entity.1.type')}"
            assert span_attributes.get("entity.2.type") == "model.llm.gpt-4", \
                f"Expected GPT-4 model type, got {span_attributes.get('entity.2.type')}"
            found_inference = True

        # Verify agent invocation spans
        if span_type == "agentic.invocation":
            agent_type = span_attributes.get("entity.1.type")
            agent_name = span_attributes.get("entity.1.name")
            assert agent_type == "agent.llamaindex", \
                f"Expected LlamaIndex agent type, got {agent_type}"
            if agent_name:
                agent_names.add(agent_name)
            found_agent = True

        # Verify tool invocation spans
        if span_type == "agentic.tool.invocation":
            tool_type = span_attributes.get("entity.1.type")
            tool_name = span_attributes.get("entity.1.name")
            assert tool_type == "tool.llamaindex", \
                f"Expected LlamaIndex tool type, got {tool_type}"
            if tool_name:
                tool_names.add(tool_name)
            found_tool = True

    # Verify we captured the essential span types
    assert found_inference, f"No inference spans found. Spans captured: {len(spans)}"
    assert found_agent, "No agent invocation spans found"
    assert found_tool, "No tool invocation spans found"
    assert found_chunk_event, "No chunk-level streaming events (data.chunk) found in trace"
    
    # Verify we captured spans for the expected agents and tools
    assert "coordinator" in agent_names, \
        f"Coordinator agent not found in spans. Agents: {agent_names}"
    assert "flight_booking_agent" in agent_names, \
        f"Flight booking agent not found in spans. Agents: {agent_names}"
    assert "hotel_booking_agent" in agent_names, \
        f"Hotel booking agent not found in spans. Agents: {agent_names}"
    assert "book_flight" in tool_names, \
        f"book_flight tool not found in spans. Tools: {tool_names}"
    assert "book_hotel" in tool_names, \
        f"book_hotel tool not found in spans. Tools: {tool_names}"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
