import pytest
from monocle_test_tools import TestCase, MonocleValidator
from test_common.crewai_travel_agent import hotel_booking_agent, hotel_tool, create_agents
from crewai import Crew, Task

# Setup telemetry instrumentation for CrewAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

# Initialize instrumentation once
#setup_monocle_telemetry(workflow_name="crewai_simple_test")

agent_test_cases: list[TestCase] = [
    {
        "test_input": ["Book a hotel room at Marriott in New York for 2 nights starting Dec 1st 2025."],
        "test_output": "Successfully booked a stay at Marriott in New York from Dec 1st 2025 for 2 nights.",
        "comparer": "similarity"
    },
    {
        "test_input": ["Book a hotel room at Marriott in New York for 2 nights starting Dec 1st 2025."],
        "test_spans": [
            {
                "span_type": "agentic.turn",
                "entities": [
                    {"type": "agent", "name": "CrewAI Travel Agent"}
                ]
            },
            {
                "span_type": "agentic.invocation",
                "entities": [
                    {"type": "agent", "name": "CrewAI Hotel Booking Agent"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation", 
                "entities": [
                    {"type": "tool", "name": "crew_book_hotel"}
                ]
            }
        ]
    }
]


async def execute_simple_hotel_booking(request: str):
    """Execute a simple hotel booking using single CrewAI agent."""
    
    # Get agents (they'll be created with OpenAI setup)
    hotel_agent, _, _ = create_agents()
    
    # Create simple task
    task = Task(
        name = "Hotel Booking Task",
        description=f"Book hotel based on this request: {request}",
        expected_output="Hotel booking confirmation",
        agent=hotel_agent
    )

    # Create simple crew with single agent and task
    crew = Crew(
        agents=[hotel_agent],
        tasks=[task],
        verbose=True
    )
    
    result = await crew.kickoff_async(inputs={
        "hotel_request": request
    })
    return str(result)


@MonocleValidator().monocle_testcase(agent_test_cases)
async def test_crewai_simple_hotel_agent(my_test_case: TestCase):# @pytest.mark.asyncio
# @pytest.mark.parametrize("test_case", agent_test_cases)
# async def test_crewai_simple_hotel_agent(monocle_test_case):
    """Test simple CrewAI hotel booking agent."""
    # Extract the hotel request from test input
    hotel_request = agent_test_cases.test_input[0]
    
    # Execute the simple hotel booking
    result = await execute_simple_hotel_booking(hotel_request)
    
    return result


if __name__ == "__main__":
    pytest.main([__file__])