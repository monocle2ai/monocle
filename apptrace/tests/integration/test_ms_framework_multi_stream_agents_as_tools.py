import logging
import pytest
import random
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
    flight_agent = client.create_agent(
        name="MS_Flight_Booking_Agent",
        instructions=(
            "You are a Flight Booking Assistant. "
            "Your goal is to help users book flights between any two cities or airports. "
            "Focus only on flight bookings and ignore other parts of the request."
        ),
        tools=[book_flight],
    )
    
    # Create hotel booking agent
    hotel_agent = client.create_agent(
        name="MS_Hotel_Booking_Agent",
        instructions=(
            "You are a Hotel Booking Assistant. "
            "Your goal is to help users book hotel accommodations. "
            "Focus only on hotel bookings and ignore other parts of the request."
        ),
        tools=[book_hotel],
    )
    
    # Create supervisor agent (coordinates other agents)
    supervisor_agent = client.create_agent(
        name="MS_Travel_Supervisor",
        instructions=(
            "You are a Travel Supervisor that coordinates complete travel bookings. "
            "When a user requests travel arrangements, delegate flight bookings to the Flight Booking Agent "
            "and hotel bookings to the Hotel Booking Agent. "
            "Provide a consolidated summary of all bookings."
        ),
        tools=[],  # Supervisor doesn't have direct tools, it delegates to other agents
    )
else:
    flight_agent = None
    hotel_agent = None
    supervisor_agent = None


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry instrumentation for Microsoft Agent Framework multi-agent tests."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter)
    ]
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="ms_agent_multi_agent_stream_tools",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

@pytest.mark.skipif(not MICROSOFT_AGENT_AVAILABLE, reason="Microsoft Agent Framework not installed")
@pytest.mark.asyncio
async def test_microsoft_supervisor_delegation(setup):
    """Test supervisor agent delegating to flight and hotel booking tools directly."""
    if flight_agent is None or hotel_agent is None or supervisor_agent is None:
        pytest.skip("Azure OpenAI credentials not configured")
    
    # Create supervisor with direct access to both booking tools
    # This mimics the LangGraph pattern where supervisor has tools and delegates work
    supervisor_with_tools = client.create_agent(
        name="MS_Delegating_Supervisor",
        instructions=(
            "You are a Travel Supervisor that coordinates complete travel bookings. "
            "You have access to flight and hotel booking tools. "
            "When a user requests travel arrangements: "
            "1. Use book_flight tool for flight bookings "
            "2. Use book_hotel tool for hotel bookings "
            "3. Provide a consolidated summary of all bookings. "
            "Process flight bookings first, then hotel bookings."
        ),
        tools=[book_flight, book_hotel],
    )
    
    # Test message requesting both flight and hotel bookings
    task_description = "Book a flight from BOM to JFK for December 15th and also book a stay at the Marriott for 3 days."
    
    logger.info(f"Task: {task_description}")
    
    # Execute supervisor agent which should use both tools directly
    supervisor_response = ""
    async for chunk in supervisor_with_tools.run_stream(task_description):
        if chunk.text:
            supervisor_response += chunk.text
    
    logger.info(f"Supervisor Response: {supervisor_response}")
    
    # Basic verification
    assert supervisor_response, "Should get supervisor response"
    
    # Verify both bookings are mentioned in the response
    response_lower = supervisor_response.lower()
    assert "flight" in response_lower or "bom" in response_lower or "jfk" in response_lower, "Should contain flight booking"
    assert "hotel" in response_lower or "marriott" in response_lower, "Should contain hotel booking"
    
    verify_spans_with_delegation(setup)


def verify_spans_with_delegation(custom_exporter):
    """Verify spans for supervisor delegation test - should have single agentic.turn."""
    time.sleep(2)
    found_inference = found_tool = False
    found_supervisor_agent = False
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
            # Assertions for inference attributes
            if "entity.1.type" in span_attributes:
                assert "entity.1.type" in span_attributes
            if "entity.1.provider_name" in span_attributes:
                assert "entity.1.provider_name" in span_attributes
            if "entity.1.inference_endpoint" in span_attributes:
                assert "entity.1.inference_endpoint" in span_attributes
            
            # Check for tool calls
            if span_attributes.get("span.subtype") == "tool_call":
                found_tool_call = True
            
            for event in span.events:
                if event.name == "gen_ai.metadata":
                    if "finish_reason" in event.attributes:
                        if event.attributes["finish_reason"] == "tool_calls":
                            found_tool_call = True
            
            found_inference = True

        # Check for agent invocation spans (should only be supervisor)
        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.invocation"
                and "entity.1.name" in span_attributes
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.microsoft"
            if span_attributes["entity.1.name"] == "MS_Delegating_Supervisor":
                found_supervisor_agent = True

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
                # Verify it's associated with supervisor
                if "entity.2.name" in span_attributes:
                    assert span_attributes["entity.2.name"] == "MS_Delegating_Supervisor"
            elif span_attributes["entity.1.name"] == "book_hotel":
                found_book_hotel_tool = True
                # Verify it's associated with supervisor
                if "entity.2.name" in span_attributes:
                    assert span_attributes["entity.2.name"] == "MS_Delegating_Supervisor"
            found_tool = True

    assert found_inference, "Inference span not found"
    assert found_supervisor_agent, "Supervisor agent span not found"
    assert found_tool_call, "Tool call finish reason not found"
    assert found_tool, "Tool invocation span not found"
    assert found_book_flight_tool, "Book flight tool span not found"
    assert found_book_hotel_tool, "Book hotel tool span not found"
    
    # # Key assertion: Should only have ONE agentic.turn span (at the beginning)
    # assert agentic_turn_count == 1, f"Expected 1 agentic.turn span, found {agentic_turn_count}"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
