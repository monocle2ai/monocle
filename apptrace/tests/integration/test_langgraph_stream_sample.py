import logging
import time

import pytest
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from langchain.agents import create_agent

logger = logging.getLogger(__name__)

coffee_menu = {
    "espresso": "A strong and bold coffee shot.",
    "latte": "A smooth coffee with steamed milk.",
    "cappuccino": "A rich coffee with frothy milk foam.",
    "americano": "Espresso with added hot water for a milder taste.",
    "mocha": "A chocolate-flavored coffee with whipped cream."
}


@pytest.fixture(scope="module")
def setup():
    memory_exporter = InMemorySpanExporter()
    span_processors = [
        BatchSpanProcessor(FileSpanExporter()),
        SimpleSpanProcessor(memory_exporter)
    ]
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="langgraph_stream_agent",
            span_processors=span_processors
        )
        yield memory_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@tool
def get_coffee_details(coffee_name: str) -> str:
    """Provides details about a specific coffee. Input the coffee name."""
    return coffee_menu.get(coffee_name.lower(), "Sorry, we don't have details for that coffee.")


def test_sync_stream_chat_sample(setup):
    """Test synchronous streaming with stream() stream_mode='values'."""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    tools = [get_coffee_details]
    agent_executor = create_agent(llm, tools)

    chunk_count = 0
    for chunk in agent_executor.stream(
        input={"messages": [HumanMessage(content="get details about a cappuccino")]},
        stream_mode="values"
    ):
        chunk_count += 1
        logger.info(f"Sync stream chunk {chunk_count}: {chunk}")

    logger.info(f"Total sync stream chunks received: {chunk_count}")
    time.sleep(2)
    verify_stream_spans(memory_exporter=setup, test_name="sync_stream_values")


@pytest.mark.asyncio(loop_scope="function")
async def test_async_stream_chat_sample(setup):
    """Test asynchronous streaming with astream stream_mode='values'."""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    tools = [get_coffee_details]
    agent_executor = create_agent(llm, tools)

    chunk_count = 0
    async for chunk in agent_executor.astream(
        input={"messages": [HumanMessage(content="get details about a latte")]},
        stream_mode="values"
    ):
        chunk_count += 1
        logger.info(f"Async stream chunk {chunk_count}: {chunk}")

    logger.info(f"Total async stream chunks received: {chunk_count}")
    time.sleep(2)
    verify_stream_spans(memory_exporter=setup, test_name="async_stream_values")


@pytest.mark.asyncio(loop_scope="function")
async def test_async_stream_updates_mode(setup):
    """Test async streaming with astream stream_mode='updates' (node-keyed chunks)."""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    tools = [get_coffee_details]
    agent_executor = create_agent(llm, tools)

    chunk_count = 0
    async for chunk in agent_executor.astream(
        input={"messages": [HumanMessage(content="get details about an espresso")]},
        stream_mode="updates"
    ):
        chunk_count += 1
        logger.info(f"Updates mode chunk {chunk_count}: {list(chunk.keys())}")

    logger.info(f"Total updates mode chunks received: {chunk_count}")
    time.sleep(2)
    verify_stream_spans(memory_exporter=setup, test_name="async_stream_updates")


@pytest.mark.asyncio(loop_scope="function")
async def test_async_stream_messages_mode(setup):
    """Test async streaming with astream stream_mode='messages' (v2 dict StreamParts)."""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    tools = [get_coffee_details]
    agent_executor = create_agent(llm, tools)

    chunk_count = 0
    async for chunk in agent_executor.astream(
        input={"messages": [HumanMessage(content="get details about a mocha")]},
        stream_mode="messages"
    ):
        chunk_count += 1
        logger.info(f"Messages mode chunk {chunk_count}: type={chunk[0] if isinstance(chunk, (list, tuple)) else type(chunk).__name__}")

    logger.info(f"Total messages mode chunks received: {chunk_count}")
    time.sleep(2)
    verify_stream_spans(memory_exporter=setup, test_name="async_stream_messages")


def verify_stream_spans(memory_exporter=None, test_name=""):
    """Verify streaming spans are correctly instrumented."""
    found_inference = found_agent = found_tool = found_turn = False
    parent_command_exceptions_found = []

    spans = memory_exporter.get_finished_spans()
    logger.info(f"[{test_name}] Total spans collected: {len(spans)}")

    for span in spans:
        span_attributes = span.attributes
        span_type = span_attributes.get("span.type")

        # Check for ParentCommand exceptions in span events
        for event in span.events:
            if hasattr(event, "attributes") and event.attributes:
                exc_type = event.attributes.get("exception.type", "")
                if "ParentCommand" in str(exc_type):
                    parent_command_exceptions_found.append({
                        "span_name": span.name,
                        "span_id": span.context.span_id,
                        "exception_type": exc_type,
                        "exception_message": event.attributes.get("exception.message", ""),
                    })

        # Inference spans
        if span_type in ("inference", "inference.framework"):
            assert span_attributes.get("entity.1.type") == "inference.openai", (
                f"Expected inference.openai, got {span_attributes.get('entity.1.type')}"
            )
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes.get("entity.2.name") == "gpt-4"
            assert span_attributes.get("entity.2.type") == "model.llm.gpt-4"

            if len(span.events) >= 3:
                _span_input, _span_output, span_metadata = span.events[:3]
                if "completion_tokens" in span_metadata.attributes:
                    found_inference = True

        if span_type == "agentic.invocation" and "entity.1.name" in span_attributes:
            assert "entity.1.type" in span_attributes
            assert span_attributes.get("entity.1.type") == "agent.langgraph"
            found_agent = True

        if span_type == "agentic.tool.invocation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes.get("entity.1.type") == "tool.langgraph"
            found_tool = True

            if len(span.events) >= 2:
                span_input, _span_output = span.events[:2]
                assert "input" in span_input.attributes
                assert span_input.attributes.get("input") != ""

        if span_type == "agentic.turn":
            assert "scope.agentic.turn" in span_attributes
            if len(span.events) >= 2:
                span_input, _span_output = span.events[:2]
                assert "input" in span_input.attributes
                assert span_input.attributes.get("input") != ""
            found_turn = True

    span_types_found = {s.attributes.get("span.type") for s in spans}
    assert found_agent, f"Agent span not found. Span types found: {span_types_found}"
    assert found_tool, f"Tool span not found. Span types found: {span_types_found}"
    assert found_turn, f"Turn span not found. Span types found: {span_types_found}"

    assert len(parent_command_exceptions_found) == 0, (
        f"Found {len(parent_command_exceptions_found)} ParentCommand exceptions that should be "
        f"suppressed. Details: {parent_command_exceptions_found}"
    )

    logger.info(f"✓ [{test_name}] Stream span verification passed")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
