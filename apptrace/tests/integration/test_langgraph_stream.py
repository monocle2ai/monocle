import logging
import time

import pytest
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from langgraph.prebuilt import create_react_agent

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
    try:
        instrumentor = setup_monocle_telemetry(
                    workflow_name="langgraph_stream_agent",
                    span_processors=span_processors
                    )
        yield memory_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

@tool
def get_coffee_details(coffee_name: str) -> str:
    """Provides details about a specific coffee. Input the coffee name."""
    coffee_name = coffee_name.lower()
    return coffee_menu.get(coffee_name, "Sorry, we don't have details for that coffee.")


# Define tools for the chatbot
tools = [get_coffee_details]

# Set up the Chat Model
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

@pytest.mark.asyncio
async def test_async_stream_chat_sample(setup):
    """Test asynchronous streaming with astream_mode='values'."""

    agent_executor = create_react_agent(llm, tools, name="coffee_agent")

    # Consume stream chunks asynchronously
    chunk_count = 0
    last_chunk = None
    async for chunk in agent_executor.astream(
        input={"messages": [HumanMessage(content="get details about a latte")]},
        stream_mode="values"
    ):
        chunk_count += 1
        last_chunk = chunk
        logger.info(f"Async stream chunk {chunk_count}: {chunk}")

    logger.info(f"Total async stream chunks received: {chunk_count}")
    time.sleep(2)
    verify_stream_spans(memory_exporter=setup, test_name="async_stream")

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
            if hasattr(event, 'attributes') and event.attributes:
                # Check if this is an exception event with ParentCommand
                if (event.attributes.get("exception.type") == "langgraph.errors.ParentCommand" or
                    "ParentCommand" in str(event.attributes.get("exception.type", ""))):
                    parent_command_exceptions_found.append({
                        "span_name": span.name,
                        "span_id": span.context.span_id,
                        "exception_type": event.attributes.get("exception.type"),
                        "exception_message": event.attributes.get("exception.message", "")
                    })

        # Check inference spans
        if span_type in ["inference", "inference.framework"]:
            # Assertions for all inference attributes
            assert span_attributes.get("entity.1.type") == "inference.openai", f"Expected inference.openai, got {span_attributes.get('entity.1.type')}"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes.get("entity.2.name") == "gpt-4"
            assert span_attributes.get("entity.2.type") == "model.llm.gpt-4"

            # Assertions for metadata
            if len(span.events) >= 3:
                span_input, span_output, span_metadata = span.events[:3]
                if "completion_tokens" in span_metadata.attributes:
                    found_inference = True

        # Check agent invocation spans
        if span_type == "agentic.invocation" and "entity.1.name" in span_attributes:
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes.get("entity.1.type") == "agent.langgraph"
            found_agent = True

        # Check tool invocation spans
        if span_type == "agentic.tool.invocation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes.get("entity.1.type") in ["tool.langgraph"]
            found_tool = True
            
            if len(span.events) >= 2:
                span_input, span_output = span.events[:2]
                assert "input" in span_input.attributes
                assert span_input.attributes.get("input") != ""

        # Check agentic turn spans
        if span_type == "agentic.turn":
            assert "scope.agentic.turn" in span_attributes
            if len(span.events) >= 2:
                span_input, span_output = span.events[:2]
                assert "input" in span_input.attributes
                assert span_input.attributes.get("input") != ""
            found_turn = True

    # Verify we found the expected spans
    assert found_agent, f"Agent span not found. Span types found: {set(s.attributes.get('span.type') for s in spans)}"
    assert found_tool, f"Tool span not found. Span types found: {set(s.attributes.get('span.type') for s in spans)}"
    assert found_turn, f"Turn span not found. Span types found: {set(s.attributes.get('span.type') for s in spans)}"
    
    # Check for ParentCommand exceptions
    assert len(parent_command_exceptions_found) == 0, (
        f"Found {len(parent_command_exceptions_found)} ParentCommand exceptions in spans. "
        f"These should be suppressed by the ParentCommand filter. Found exceptions: {parent_command_exceptions_found}"
    )
    
    logger.info(f"✓ [{test_name}] Stream span verification passed")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
