import logging
import pytest
import time
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

# Import Microsoft Agent Framework components
try:
    from agent_framework.azure import AzureOpenAIChatClient
    from azure.identity.aio import AzureCliCredential
    from typing import Annotated
    import os
    MICROSOFT_AGENT_AVAILABLE = True
except ImportError:
    MICROSOFT_AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)


# Simple tool - flight booking
def book_flight(
    from_airport: Annotated[str, "The departure airport code (e.g., JFK, LAX)"],
    to_airport: Annotated[str, "The destination airport code (e.g., SFO, ORD)"],
) -> str:
    """Book a flight from one airport to another"""
    import random
    
    # Simple booking simulation
    confirmation = f"FL{random.randint(100000, 999999)}"
    
    return f"Successfully booked flight #{confirmation} from {from_airport} to {to_airport}."


# Check for required environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") if MICROSOFT_AGENT_AVAILABLE else None
deployment = os.getenv("AZURE_OPENAI_API_DEPLOYMENT") if MICROSOFT_AGENT_AVAILABLE else None

# Initialize Azure OpenAI client and agent at module level
if MICROSOFT_AGENT_AVAILABLE and endpoint and deployment:
    client = AzureOpenAIChatClient(
        endpoint=endpoint,
        deployment_name=deployment,
        credential=AzureCliCredential(),
    )
    
    # Create agent with tool
    agent = client.create_agent(
        name="MS_Flight_Booking_Agent",
        instructions=(
            "You are a Flight Booking Assistant. "
            "Your goal is to help users book flights between any two cities or airports. "
            "You are a reliable flight booking assistant helping users plan their air travel efficiently."
        ),
        tools=[book_flight],
    )
    print(f"🔍 Agent class: {type(agent).__name__}")
    print(f"🔍 Agent module: {type(agent).__module__}")
else:
    agent = None


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry instrumentation for Microsoft Agent Framework tests."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter)
    ]
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="microsoft_agent_simple_test",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.mark.skipif(not MICROSOFT_AGENT_AVAILABLE, reason="Microsoft Agent Framework not installed")
@pytest.mark.asyncio
async def test_microsoft_agent_simple(setup):
    """Simple Microsoft Agent Framework test with 1 agent and 1 tool."""
    if agent is None:
        pytest.skip("Azure OpenAI credentials not configured")
    
    # Execute the agent with simple task
    task_description = "Book a flight from JFK to SFO for the user."
    
    logger.info(f"Task: {task_description}")
    
    # Run the agent and collect response
    response_text = ""
    async for chunk in agent.run_stream(task_description):
        if chunk.text:
            response_text += chunk.text
    
    logger.info(f"Result: {response_text}")
    
    # Basic verification
    assert response_text, "Should get a response"
    verify_spans(setup)


def verify_spans(custom_exporter):
    time.sleep(2)
    found_inference = found_agent = found_tool = found_tool_call = False
    spans = custom_exporter.get_captured_spans()
    
    for span in spans:
        span_attributes = span.attributes

        # Check for inference spans
        if "span.type" in span_attributes and (
                span_attributes["span.type"] == "inference"
                or span_attributes["span.type"] == "inference.framework"
                or span_attributes["span.type"] == "inference.modelapi"
        ):
            # Assertions for inference attributes
            # Note: inference.modelapi spans may not have all entity attributes
            if "entity.1.type" in span_attributes:
                assert "entity.1.type" in span_attributes
            if "entity.1.provider_name" in span_attributes:
                assert "entity.1.provider_name" in span_attributes
            if "entity.1.inference_endpoint" in span_attributes:
                assert "entity.1.inference_endpoint" in span_attributes
            
            # Check for tool calls in metadata or subtype
            if span_attributes.get("span.subtype") == "tool_call":
                found_tool_call = True
            
            for event in span.events:
                if event.name == "gen_ai.metadata":
                    if "finish_reason" in event.attributes:
                        if event.attributes["finish_reason"] == "tool_calls":
                            found_tool_call = True
            
            found_inference = True

        # Check for agent invocation spans
        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.invocation"
                and "entity.1.name" in span_attributes
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.microsoft"
            if span_attributes["entity.1.name"] == "MS_Flight_Booking_Agent":
                found_agent = True

        # Check for tool invocation spans
        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.microsoft"
            if span_attributes["entity.1.name"] == "book_flight":
                found_tool = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool_call, "Tool call finish reason not found"
    assert found_tool, "Tool invocation span not found"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
