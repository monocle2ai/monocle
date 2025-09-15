from unittest import mock
import pytest 
import logging
from monocle_test_tools.test_tools.validator import MonocleValidator, TestCase, ToolToVerify, InferenceToVerify

logging.basicConfig(level=logging.WARN)

agent_test_cases = [
    TestCase(
        test_input="Book a flight from San Francisco to Mumbai, book Taj Mahal hotel in Mumbai.",
        expected_response="Flight is booked from San Francisco to Mumbai. Stay is booked at Taj Mahal hotel in Mumbai.",
    ),
    TestCase(
        test_input="Book a flight from San Francisco to Mumbai for tomorrow.",
        expected_response="Flight is booked from San Francisco to Mumbai.",
    ),
    TestCase(
        tools_to_verify= [
            ToolToVerify(tool_name="adk_book_flight"),
            ToolToVerify(tool_name="adk_book_hotel", agent_name="adk_hotel_booking_agent", expected_tool_output="{'status': 'success', 'message': 'Successfully booked a stay at Marriot Intercontinental in Mumbai.'}")
        ],
        check_warnings=True
    ),
    TestCase(
        test_description="Test the end-to-end travel agent application.",
        test_input="Book a flight from San Francisco to Mumbai, book Taj Mahal hotel in Mumbai.",
        inferences_to_verify=[
            InferenceToVerify(expected_response= [
                                "{\"model\": \"I am sorry, I need the current date to book the flight for tomorrow. Could you please provide the current date?\\n\"}",
                                "{\"model\": \"I am sorry, I am designed to only book hotels.\"}",
                                "{\"model\": \"Okay, I will summarize the travel details once the flight booking agent has provided the flight information. Currently, I am waiting for the current date from the flight booking agent to book the flight from SFO to JFK for tomorrow.\\n\"}"
                                ]
            )
        ]
    )
]

@mock
def run_agent(test_input: str):
    # Placeholder for actual agent invocation logic
    return "Mocked agent response"

@MonocleValidator().monocle_testcase(agent_test_cases)
async def test_agents(my_test_case: TestCase):
    await run_agent(my_test_case.test_input)

if __name__ == "__main__":
    pytest.main([__file__]) 