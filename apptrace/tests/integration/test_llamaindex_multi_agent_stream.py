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
    )

    return AgentWorkflow(
        agents=[coordinator, flight_agent, hotel_agent],
        root_agent=coordinator.name,
    )


async def run_streaming_agent() -> tuple[str, object]:
    agent_workflow = setup_agents()

    handler_or_coro = agent_workflow.run(
        user_msg="book a flight from BOS to JFK and book a hotel stay at McKittrick Hotel"
    )
    handler = (
        await handler_or_coro
        if asyncio.iscoroutine(handler_or_coro)
        else handler_or_coro
    )

    streamed_text = []
    async for event in handler.stream_events():
        if isinstance(event, AgentStream) and event.delta:
            streamed_text.append(event.delta)

    result = await handler
    return "".join(streamed_text), result


def test_multi_agent_streaming(setup):
    streamed_text, final_result = asyncio.run(run_streaming_agent())

    assert final_result is not None, "Agent workflow should return a final response"
    assert streamed_text.strip(), "Expected non-empty streaming output from AgentStream events"

    verify_spans(memory_exporter=setup)


def verify_spans(memory_exporter=None):
    time.sleep(2)
    found_inference = False
    found_agent = False
    found_tool = False
    found_flight_agent = False
    found_hotel_agent = False
    found_supervisor_agent = False
    found_book_hotel_tool = False
    found_book_flight_tool = False
    found_book_flight_delegation = False
    found_book_hotel_delegation = False

    spans = memory_exporter.get_finished_spans()
    for span in spans:
        span_attributes = span.attributes
        span_type = span_attributes.get("span.type")

        if span_type in ("inference", "inference.framework"):
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4"
            found_inference = True

        if span_type == "agentic.invocation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.llamaindex"
            if span_attributes["entity.1.name"] == "flight_booking_agent":
                found_flight_agent = True
            elif span_attributes["entity.1.name"] == "hotel_booking_agent":
                found_hotel_agent = True
            elif span_attributes["entity.1.name"] == "coordinator":
                found_supervisor_agent = True
            found_agent = True

        if span_type == "agentic.delegation":
            assert span_attributes.get("entity.1.type") == "agent.llamaindex"
            from_agent = span_attributes.get("entity.1.from_agent")
            to_agent = span_attributes.get("entity.1.to_agent")
            if from_agent == "coordinator" and to_agent == "flight_booking_agent":
                found_book_flight_delegation = True
            elif from_agent == "flight_booking_agent" and to_agent == "hotel_booking_agent":
                found_book_hotel_delegation = True

        if span_type == "agentic.tool.invocation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert "entity.2.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.llamaindex"
            if (
                span_attributes["entity.1.name"] == "book_flight"
                and span_attributes["entity.2.name"] == "flight_booking_agent"
            ):
                found_book_flight_tool = True
            elif (
                span_attributes["entity.1.name"] == "book_hotel"
                and span_attributes["entity.2.name"] == "hotel_booking_agent"
            ):
                found_book_hotel_tool = True
            found_tool = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    assert found_flight_agent, "Flight booking agent span not found"
    assert found_hotel_agent, "Hotel booking agent span not found"
    assert found_supervisor_agent, "Coordinator agent span not found"
    assert found_book_flight_tool, "Book flight tool span not found"
    assert found_book_hotel_tool, "Book hotel tool span not found"
    assert found_book_flight_delegation, "Book flight delegation span not found"
    assert found_book_hotel_delegation, "Book hotel delegation span not found"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
