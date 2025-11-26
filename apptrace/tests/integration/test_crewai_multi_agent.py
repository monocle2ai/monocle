import logging
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

# Import CrewAI components
try:
    from crewai import Agent, Crew, Task
    from crewai.tools import BaseTool
    from langchain_openai import ChatOpenAI
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

logger = logging.getLogger(__name__)


# Hotel booking tool
class BookHotelTool(BaseTool):
    name: str = "book_hotel"
    description: str = "Book a hotel reservation"

    def _run(self, hotel_name: str, city: str, nights: int = 1) -> str:
        import time
        import random
        
        # Simple booking simulation
        time.sleep(0.1)
        confirmation = f"HT{random.randint(100000, 999999)}"
        cost = nights * 150
        
        return f"HOTEL BOOKING CONFIRMED #{confirmation}: {hotel_name} in {city} for {nights} nights - ${cost}"


# Flight booking tool
class BookFlightTool(BaseTool):
    name: str = "book_flight"
    description: str = "Book a flight reservation"

    def _run(self, departure: str, arrival: str, date: str = "next week") -> str:
        import time
        import random
        
        # Simple booking simulation
        time.sleep(0.1)
        confirmation = f"FL{random.randint(100000, 999999)}"
        cost = random.randint(300, 800)
        
        return f"FLIGHT BOOKING CONFIRMED #{confirmation}: {departure} to {arrival} on {date} - ${cost}"


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry instrumentation for CrewAI multi-agent tests."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter)
    ]

    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="crewai_multi_agent_test",
            span_processors=span_processors,
        )
        yield file_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def create_multi_agent_crew():
    """Create a multi-agent CrewAI crew with 2 agents, 2 tools, and 1 supervisor."""
    if not CREWAI_AVAILABLE:
        pytest.skip("CrewAI not available")

    # Create tools
    hotel_tool = BookHotelTool()
    flight_tool = BookFlightTool()

    # Create hotel booking agent
    hotel_agent = Agent(
        role="Hotel Booking Agent",
        goal="Book the best hotel accommodations for travelers",
        backstory="You are an expert hotel booking agent with extensive knowledge of hotels worldwide.",
        tools=[hotel_tool],
        llm=ChatOpenAI(model="gpt-4.1"),
        verbose=True,
        allow_delegation=False
    )

    # Create flight booking agent
    flight_agent = Agent(
        role="Flight Booking Agent", 
        goal="Book the best flight options for travelers",
        backstory="You are an expert flight booking agent with access to all major airlines.",
        tools=[flight_tool],
        llm=ChatOpenAI(model="gpt-4.1"),
        verbose=True,
        allow_delegation=False
    )

    # Create supervisor agent
    supervisor_agent = Agent(
        role="Travel Supervisor",
        goal="Coordinate complete travel bookings by delegating to agents",
        backstory="You are a travel supervisor who manages a team of booking agents. You delegate hotel bookings to the hotel agent and flight bookings to the flight agent.",
        llm=ChatOpenAI(model="gpt-4.1"),
        verbose=True,
        allow_delegation=False
    )

    # Create tasks
    hotel_task = Task(
        name="Hotel Booking Task",
        description="Book a hotel room at Marriott in New York for 2 nights.",
        expected_output="Hotel booking confirmation with confirmation number and cost",
        agent=hotel_agent
    )

    flight_task = Task(
        name = "Flight Booking Task",
        description="Book a flight from Los Angeles to New York for next week.",
        expected_output="Flight booking confirmation with confirmation number and cost",
        agent=flight_agent
    )

    supervisor_task = Task(
        name = "Travel Booking Supervision Task",
        description="Coordinate a complete travel package: book both flight from Los Angeles to New York and hotel accommodation at Marriott in New York for 2 nights. Ensure both bookings are completed successfully.",
        expected_output="Complete travel booking summary with both flight and hotel confirmations",
        agent=supervisor_agent
    )

    # Create crew with all agents and tasks
    # In sequential mode, the last task determines the final output
    # So we put supervisor task last to get the complete summary
    crew = Crew(
        agents=[supervisor_agent, hotel_agent, flight_agent],
        tasks=[hotel_task, flight_task, supervisor_task],
        verbose=True,
        process="sequential"  # Execute tasks in sequence
    )
    return crew


@pytest.mark.skipif(not CREWAI_AVAILABLE, reason="CrewAI not installed")
def test_crewai_multi_agent(setup):
    """Multi-agent CrewAI test with 2 agents, 2 tools, and 1 supervisor."""
    crew = create_multi_agent_crew()
    
    # Execute the crew with travel booking request
    result = crew.kickoff(inputs={
        "travel_request": "Plan a complete trip from Los Angeles to New York with hotel stay"
    })
    logger.info(f"Multi-agent result: {result}")
    
    # Basic verification
    assert result is not None, "Should get a result"
    
    # Verify both bookings are mentioned in result
    result_str = str(result)
    assert "HOTEL BOOKING CONFIRMED" in result_str or "hotel" in result_str.lower(), "Should contain hotel booking"
    assert "FLIGHT BOOKING CONFIRMED" in result_str or "flight" in result_str.lower(), "Should contain flight booking"
    
    logger.info("✓ Multi-agent CrewAI test completed")


@pytest.mark.skipif(not CREWAI_AVAILABLE, reason="CrewAI not installed") 
@pytest.mark.asyncio
async def test_crewai_multi_agent_async(setup):
    """Async multi-agent CrewAI test."""
    crew = create_multi_agent_crew()
    
    # Execute the crew asynchronously
    result = await crew.kickoff_async(inputs={
        "travel_request": "Plan a complete trip from Los Angeles to New York with hotel stay"
    })
    logger.info(f"Async multi-agent result: {result}")
    
    # Basic verification
    assert result is not None, "Should get a result"
    
    logger.info("✓ Async multi-agent CrewAI test completed")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])