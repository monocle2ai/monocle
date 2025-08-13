import asyncio
import pytest
import logging
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
import time

logger = logging.getLogger(__name__)

memory_exporter = InMemorySpanExporter()
span_processors = [
    SimpleSpanProcessor(memory_exporter),
    BatchSpanProcessor(FileSpanExporter()),
]

@pytest.fixture(scope="module")
def setup():
    memory_exporter.clear()
    setup_monocle_telemetry(
        workflow_name="agents_sdk_test",
        span_processors=span_processors,
    )

def book_hotel(hotel_name: str) -> str:
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name} for 50 USD."

def book_flight(from_airport: str, to_airport: str) -> str:
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport} for 100 USD."

def get_weather(city: str) -> str:
    """Get weather information for a city"""
    return f"The weather in {city} is sunny and 75°F."

@pytest.mark.integration()
@pytest.mark.asyncio
async def test_agents_sdk_multi_agent(setup):
    """Test multi-agent interaction with handoffs."""
    try:
        from agents import Agent, Runner, function_tool
        
        # Create tools
        @function_tool
        def book_flight_tool(from_airport: str, to_airport: str) -> str:
            """Book a flight between airports."""
            return book_flight(from_airport, to_airport)
        
        @function_tool
        def book_hotel_tool(hotel_name: str) -> str:
            """Book a hotel reservation."""
            return book_hotel(hotel_name)
        
        @function_tool
        def get_weather_tool(city: str) -> str:
            """Get weather information."""
            return get_weather(city)
        
        # Create specialized agents
        flight_agent = Agent(
            name="Flight Agent",
            instructions="You are a flight booking specialist. Use the book_flight_tool to book flights.",
            tools=[book_flight_tool, get_weather_tool]
        )
        
        hotel_agent = Agent(
            name="Hotel Agent", 
            instructions="You are a hotel booking specialist. Use the book_hotel_tool to book hotels.",
            tools=[book_hotel_tool]
        )
        
        # Create a coordinator agent with handoffs
        coordinator = Agent(
            name="Travel Coordinator",
            instructions="You are a travel coordinator. Delegate flight bookings to the Flight Agent and hotel bookings to the Hotel Agent.",
            handoffs=[flight_agent, hotel_agent],
            tools=[get_weather_tool]
        )
        
        # Test the multi-agent workflow
        result = await Runner.run(
            coordinator,
            "I need to book a flight from NYC to LAX and also book the Hilton hotel in Los Angeles. Also check the weather in Los Angeles."
        )
        
        print(f"Multi-agent result: {result.final_output}")
        
        # Verify spans were created
        verify_multi_agent_spans()
        
    except ImportError:
        pytest.skip("OpenAI Agents SDK not available")

def verify_multi_agent_spans():
    """Verify that multi-agent spans were created."""
    time.sleep(2)  
    # Allow time for spans to be processed
    
    found_agent = found_tool = found_delegation = False
    agent_names = set()
    tool_names = set()
    
    spans = memory_exporter.get_finished_spans()
    
    for span in spans:
        span_attributes = span.attributes
        
        # Check for agent spans
        if ("span.type" in span_attributes and 
            span_attributes["span.type"] == "agentic.invocation"):
            assert span_attributes["entity.1.type"] == "agent.openai_agents"
            assert "entity.1.name" in span_attributes
            agent_names.add(span_attributes["entity.1.name"])
            found_agent = True
        
        # Check for tool spans
        if ("span.type" in span_attributes and 
            span_attributes["span.type"] == "agentic.tool.invocation"):
            assert span_attributes["entity.1.type"] == "tool.openai_agents"
            assert "entity.1.name" in span_attributes
            tool_names.add(span_attributes["entity.1.name"])
            found_tool = True
        
        # Check for delegation spans
        if ("span.type" in span_attributes and 
            span_attributes["span.type"] == "agentic.delegation"):
            assert span_attributes["entity.1.type"] == "agent.openai_agents"
            assert "entity.1.from_agent" in span_attributes
            assert "entity.1.to_agent" in span_attributes
            found_delegation = True
    
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    # Note: Delegation might not always occur depending on the model's decisions
    
    print(f"Found agents: {agent_names}")
    print(f"Found tools: {tool_names}")

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
