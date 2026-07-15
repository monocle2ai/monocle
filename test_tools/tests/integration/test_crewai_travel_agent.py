from asyncio import sleep
import pytest

from monocle_test_tools import TestCase, MonocleValidator
from test_common.crewai_travel_agent import execute_crewai_travel_request

agent_test_cases: list[TestCase] = [
    # {
    #     "test_input": ["Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights."],
    #     "test_output": "A flight from San Francisco to Mumbai on November 26, 2025, and a four-night stay at the Marriot Intercontinental in Juhu, Mumbai starting November 27, 2025, have been booked.",
    #     "comparer": "similarity"
    # },
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights."],
        "test_spans": [
            {
                "span_type": "agentic.turn",
                "output": "A flight from San Francisco to Mumbai on November 26, 2025, and a four-night stay at the Marriot Intercontinental in Juhu, Mumbai starting November 27, 2025, have been booked.",
                "comparer": "similarity"
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "crew_book_hotel"},
                    {"type": "agent", "name": "CrewAI Hotel Booking Agent"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "crew_book_flight"},
                    {"type": "agent", "name": "CrewAI Flight Booking Agent"}
                ]
            }
        ]
    },
    {
        "test_input": ["Book a hotel room at Marriott in New York for 2 nights starting Dec 1st 2025."],
        "test_spans": [
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
    },
    {
        "test_input": ["Book a flight from Los Angeles to New York for next Friday."],
        "test_spans": [
            {
                "span_type": "agentic.invocation", 
                "entities": [
                    {"type": "agent", "name": "CrewAI Flight Booking Agent"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "crew_book_flight"}
                ]
            }
        ]
    }
]


@MonocleValidator().monocle_testcase(agent_test_cases)
async def test_crewai_travel_agent(my_test_case: TestCase):
    """Test CrewAI travel agent with monocle test tools."""
    # Extract the travel request from test input
    travel_request = my_test_case.test_input[0]
    
    # Execute the CrewAI travel request
    result = await execute_crewai_travel_request(travel_request)
    
    # Return the result for validation
    await sleep(2)  # To avoid rate limiting
    return result


if __name__ == "__main__":
    pytest.main([__file__])