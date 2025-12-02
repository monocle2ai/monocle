from asyncio import sleep
import pytest

from monocle_test_tools import MonocleValidator
from test_common.adk_travel_agent import root_agent

agent_test_cases:list[dict] = [
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights."],
        "test_output": "A flight from San Francisco to Mumbai on November 26, 2025, and a four-night stay at the Marriot Intercontinental in Juhu, Mumbai starting November 27, 2025, have been booked.",
        "comparer": "similarity"
    },
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights."],
        "test_spans": [
            {
            "span_type": "agentic.request",
            "output": "A flight from San Francisco to Mumbai on November 26, 2025, and a four-night stay at the Marriot Intercontinental in Juhu, Mumbai starting November 27, 2025, have been booked.",
            "comparer": "similarity"
            },
            {
            "span_type": "agentic.tool.invocation",
            "entities": [
                {"type": "tool", "name": "adk_book_hotel_5"},
                {"type": "agent", "name": "adk_hotel_booking_agent_5"}
                ]
            }
        ]
    },
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights."],
        "test_spans": [
            {
            "span_type": "agentic.request",
            "eval":
                {
                "eval": "bert_score",
                "args" : [
                    "input", "output"
                ],
                "expected_result": {"Precision": 0.5, "Recall": 0.5, "F1": 0.5},
                "comparer": "metric"
                }
            }
        ]
    },
]

@MonocleValidator().monocle_testcase(agent_test_cases)
async def test_run_agents(my_test_case: dict):
   await MonocleValidator().test_agent_async(root_agent, "google_adk", my_test_case)
   await sleep(2)  # To avoid rate limiting

if __name__ == "__main__":
    pytest.main([__file__]) 