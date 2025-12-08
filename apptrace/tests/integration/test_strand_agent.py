
import asyncio
import time, os
import logging
import dotenv
import pytest
import subprocess
import sys

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
memory_exporter = InMemorySpanExporter()
@pytest.fixture(scope="module")
def setup():
    subprocess.check_call([sys.executable, "-m", "pip", "install", ".[dev_strands]"])
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="strand_agent_workflow",
            span_processors=[SimpleSpanProcessor(memory_exporter)],
        )
        yield
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

# module cleanup function
@pytest.fixture(scope="module", autouse=True)
def cleanup_module():
    yield
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y","strands-agents"])
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y","strands-agents-tools"])

def test_strand_agent(setup):
    import boto3
    from strands import Agent
    from strands import tool, tools
    from strands.models.bedrock import BedrockModel

    @tool
    def book_hotel(hotel_name: str):
        """Book a hotel"""
        return f"Successfully booked a stay at {hotel_name}."

    @tool
    def book_flight(from_airport: str, to_airport: str):
        """Book a flight"""
        return f"Successfully booked a flight from {from_airport} to {to_airport}."

#    boto_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
    boto_session = boto3.Session()
    model = BedrockModel(boto_session=boto_session, streaming=False,)

    travel_agent = Agent(name="travel_booking_agent", model=model,
                    system_prompt= """You are an agent who manages the flight and hotel booking agents.
                        Provide consolidated response to the user. Be very concise in your response.""",
                    tools = [book_flight, book_hotel],
                    description="Travel booking agent",
                )
    response = travel_agent ("Book a flight from New York to San Francisco for 26 Nov 2025.")
    verify_spans()

def verify_spans():
    time.sleep(2)
    found_inference = found_agent = found_tool = False
    found_travel_agent = False
    spans = memory_exporter.get_finished_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
                span_attributes["span.type"] == "inference"
                or span_attributes["span.type"] == "inference.framework"
        ):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.aws_bedrock"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            bedrock_model = os.getenv("BEDROCK_MODEL", "")
            assert span_attributes["entity.2.name"] == bedrock_model
            assert span_attributes["entity.2.type"] == f"model.llm.{bedrock_model}"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
            found_inference = True

            # Validate input data and output data
            input_length = len(span_input.attributes["input"])
            assert input_length > 0
            output_length = len(span_output.attributes["response"])
            assert output_length > 0

        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.invocation"
                and "entity.1.name" in span_attributes
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.strands"
            if span_attributes["entity.1.name"] == "travel_booking_agent":
                found_travel_agent = True
            found_agent = True
            # Validate input data and output data
            span_input, span_output = span.events
            assert len(span_input.attributes["input"]) > 0
            assert len(span_output.attributes["response"]) > 0

        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] in ["tool.mcp", "tool.strands"]
            if span_attributes["entity.1.name"] == "book_flight":
                found_book_flight_tool = True

            found_tool = True
            # Validate input data and output data
            span_input, span_output = span.events
            assert len(span_input.attributes["input"]) > 0
            assert len(span_output.attributes["response"]) > 0


    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    assert found_travel_agent, "Travel assistant agent span not found"
    assert found_book_flight_tool, "Book flight tool span not found"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
