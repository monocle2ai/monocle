import logging
import pytest
import random
import time
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from agent_framework import SequentialBuilder
try:
    from agent_framework.azure import AzureOpenAIChatClient
    from azure.identity.aio import AzureCliCredential
    from typing import Annotated
    import os
    MICROSOFT_AGENT_AVAILABLE = True
except ImportError:
    MICROSOFT_AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)


# Flight booking tool
def book_flight(
    from_airport: Annotated[str, "The departure airport code (e.g., JFK, LAX)"],
    to_airport: Annotated[str, "The destination airport code (e.g., SFO, ORD)"],
) -> str:
    """Book a flight from one airport to another"""
    confirmation = f"FL{random.randint(100000, 999999)}"
    cost = random.randint(300, 800)
    return f"FLIGHT BOOKING CONFIRMED #{confirmation}: {from_airport} to {to_airport} - ${cost}"


# Hotel booking tool
def book_hotel(
    hotel_name: Annotated[str, "The name of the hotel to book"],
    city: Annotated[str, "The city where the hotel is located"],
    nights: Annotated[int, "Number of nights to stay"] = 1,
) -> str:
    """Book a hotel reservation"""
    confirmation = f"HT{random.randint(100000, 999999)}"
    cost = nights * 150
    return f"HOTEL BOOKING CONFIRMED #{confirmation}: {hotel_name} in {city} for {nights} nights - ${cost}"

# Check for required environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") if MICROSOFT_AGENT_AVAILABLE else None
deployment = os.getenv("AZURE_OPENAI_API_DEPLOYMENT") if MICROSOFT_AGENT_AVAILABLE else None

# Initialize Azure OpenAI client and agents at module level
if MICROSOFT_AGENT_AVAILABLE and endpoint and deployment:
    client = AzureOpenAIChatClient(
        endpoint=endpoint,
        deployment_name=deployment,
        credential=AzureCliCredential(),
    )
    
    # Create flight booking agent
    flight_agent = client.as_agent(
        name="MS_Flight_Booking_Agent",
        instructions=(
            "You are a Flight Booking Assistant. "
            "Your goal is to help users book flights between any two cities or airports. "
            "Book the requested flight and provide confirmation details."
        ),
        tools=[book_flight],
    )
    
    # Create hotel booking agent
    hotel_agent = client.as_agent(
        name="MS_Hotel_Booking_Agent",
        instructions=(
            "You are a Hotel Booking Assistant. "
            "Your goal is to help users book hotel accommodations. "
            "Book the requested hotel and provide confirmation details."
        ),
        tools=[book_hotel],
    )
    
    # Create summarizer agent that reviews both bookings
    summarizer_agent = client.as_agent(
        name="MS_Travel_Summarizer",
        instructions=(
            "You are a Travel Booking Summarizer. "
            "Review all the booking confirmations provided and create a consolidated summary "
            "with all confirmation numbers and total costs. "
            "Provide a friendly final message to the user with all booking details."
        ),
        tools=[],
    )

    # Create sequential workflow: flight -> hotel -> summarizer
    workflow = (
        SequentialBuilder()
        .register_participants([
            lambda: flight_agent, 
            lambda: hotel_agent, 
            lambda: summarizer_agent
        ])
        .build()
    )
    
else:
    flight_agent = None
    hotel_agent = None
    summarizer_agent = None
    workflow = None


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry instrumentation for Microsoft Agent Framework sequential workflow tests."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter)
    ]
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="microsoft_agent_sequential_test",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

@pytest.mark.skipif(not MICROSOFT_AGENT_AVAILABLE, reason="Microsoft Agent Framework not installed")
@pytest.mark.asyncio
async def test_microsoft_sequential_workflow(setup):
    """Test sequential workflow where agents execute one after another."""
    if flight_agent is None or hotel_agent is None or summarizer_agent is None or workflow is None:
        pytest.skip("Azure OpenAI credentials not configured")
    
    # Test message requesting both flight and hotel bookings
    task_description = "Book a flight from BOM to JFK for December 15th and also book a stay at the Marriott for 3 days."
    
    logger.info(f"Task: {task_description}")
    
    # Execute sequential workflow
    workflow_response = await workflow.run(task_description)
    
    logger.info(f"Workflow Response: {workflow_response}")
    
    # Basic verification
    assert workflow_response, "Should get workflow response"
    
    # Verify both bookings are mentioned in the response
    response_str = str(workflow_response).lower()
    assert "flight" in response_str or "bom" in response_str or "jfk" in response_str, "Should contain flight booking"
    assert "hotel" in response_str or "marriott" in response_str, "Should contain hotel booking"
    
    verify_spans_sequential(setup)

def verify_spans_sequential(custom_exporter):
    """Verify spans for sequential workflow test."""
    time.sleep(2)
    found_inference = found_tool = False
    found_flight_agent = found_hotel_agent = found_summarizer_agent = False
    found_book_hotel_tool = found_book_flight_tool = False
    found_tool_call = False
    agentic_turn_count = 0
    
    spans = custom_exporter.get_captured_spans()
    
    for span in spans:
        span_attributes = span.attributes

        # Count agentic.turn spans
        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.turn":
            agentic_turn_count += 1

        # Check for inference spans
        if "span.type" in span_attributes and (
                span_attributes["span.type"] == "inference"
                or span_attributes["span.type"] == "inference.framework"
                or span_attributes["span.type"] == "inference.modelapi"
        ):
            # Check for tool calls
            if span_attributes.get("span.subtype") == "tool_call":
                found_tool_call = True
            
            for event in span.events:
                if event.name == "metadata":
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
            
            agent_name = span_attributes["entity.1.name"]
            if agent_name == "MS_Flight_Booking_Agent":
                found_flight_agent = True
            elif agent_name == "MS_Hotel_Booking_Agent":
                found_hotel_agent = True
            elif agent_name == "MS_Travel_Summarizer":
                found_summarizer_agent = True

        # Check for tool invocation spans
        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.microsoft"
            
            if span_attributes["entity.1.name"] == "book_flight":
                found_book_flight_tool = True
                # Verify it's associated with flight agent
                if "entity.2.name" in span_attributes:
                    assert span_attributes["entity.2.name"] == "MS_Flight_Booking_Agent"
            elif span_attributes["entity.1.name"] == "book_hotel":
                found_book_hotel_tool = True
                # Verify it's associated with hotel agent
                if "entity.2.name" in span_attributes:
                    assert span_attributes["entity.2.name"] == "MS_Hotel_Booking_Agent"
            
            found_tool = True

    assert found_inference, "Inference span not found"
    assert found_flight_agent, "Flight agent span not found"
    assert found_hotel_agent, "Hotel agent span not found"
    assert found_summarizer_agent, "Summarizer agent span not found"
    assert found_tool_call, "Tool call finish reason not found"
    assert found_tool, "Tool invocation span not found"
    assert found_book_flight_tool, "Book flight tool span not found"
    assert found_book_hotel_tool, "Book hotel tool span not found"
    
    # Should only have ONE agentic.turn span (at the workflow level)
    assert agentic_turn_count == 1, f"Expected 1 agentic.turn span, found {agentic_turn_count}"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
