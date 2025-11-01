import pytest 

from monocle_test_tools import TestCase, MonocleValidator
from test_common.adk_travel_agent import root_agent

agent_test_cases:list[TestCase] = [
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th Nov 2025."],
        "mock_tools": [
            {
                "name": "adk_book_flight_5",
                "type": "tool.adk",
                "raise_error": False,
                "error_message": "Simulated tool failure",
                "response": {
                    "status": "success",
                    "message": "Flight booked from {{from_airport}} to {{to_airport}}."
                }
            },
        ],
        "test_spans": [
            {
            "span_type": "agentic.tool.invocation",
            "entities": [
                {"type": "tool", "name": "adk_book_flight_5"},
                ],
            "output": "{'status': 'success', 'message': 'Flight booked from San Francisco to Mumbai.'}"
            }
        ]
    }
  ]

@MonocleValidator().monocle_testcase(agent_test_cases)
async def test_run_agents(my_test_case: TestCase):
   await MonocleValidator().test_agent_async(root_agent, "google_adk", my_test_case)

if __name__ == "__main__":
    pytest.main([__file__]) 