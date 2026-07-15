from asyncio import sleep
import pytest

from monocle_test_tools import TraceAssertion
from test_common.crewai_travel_agent import create_crewai_travel_crew


@pytest.mark.asyncio
async def test_tool_invocation(monocle_trace_asserter:TraceAssertion):
    """Test simple CrewAI hotel booking agent."""
    # Create crew for hotel booking - the function will create the task automatically
    travel_request = "Book a hotel room at Marriott in New York for 2 nights starting Dec 1st 2025"
    crew = create_crewai_travel_crew(travel_request)
    
    # Run the agent
    await monocle_trace_asserter.run_async_agent(crew, "crewai", travel_request)
    
    # Verify tool was called
    monocle_trace_asserter.called_tool("crew_book_hotel", "CrewAI Hotel Booking Agent").contains_output("success")

@pytest.mark.asyncio
async def test_agent_invocation(monocle_trace_asserter:TraceAssertion):
    """Test simple CrewAI hotel booking agent."""
    # Create crew for hotel booking - the function will create the task automatically
    travel_request = "Book a hotel room at Sheraton in Australia for 3 nights starting Dec 15th 2025"
    crew = create_crewai_travel_crew(travel_request)
    
    # Run the agent
    await monocle_trace_asserter.run_async_agent(crew, "crewai", travel_request)
    
    # Verify agent was invoked
    monocle_trace_asserter.called_agent("CrewAI Hotel Booking Agent").contains_output("success")

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
