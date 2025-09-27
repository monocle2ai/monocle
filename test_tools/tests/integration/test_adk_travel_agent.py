import os
import pytest 
import logging

from adk_travel_agent import run_agent, root_agent
from monocle_test_tools import TestCase, MonocleValidator
logging.basicConfig(level=logging.WARN)

agent_test_cases:list[TestCase] = [
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights."],
        "test_output": "A flight from San Francisco to Mumbai has been booked, along with a four night stay in a two queen room at the Marriot Intercontinental in Juhu, Mumbai, starting November 27th, 2025.",
        "comparer": "similarity",
        "test_spans": [
            {
            "span_type": "agentic.request",
            "output": "A flight from San Francisco to Mumbai has been booked, along with a four night stay in a two queen room at the Marriot Intercontinental in Juhu, Mumbai, starting November 27th, 2025.",
            }
        ]
    },
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights."],
        "test_spans": [
            {
            "span_type": "agentic.request",
            "output": "A flight from San Francisco to Mumbai has been booked, along with a four-night stay in a two queen room at the Marriot Intercontinental in Juhu, Mumbai, starting November 27th, 2025.",
            "comparer": "similarity",
            }
        ]
    },
    {
        "test_input": ["Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights."],
        "test_spans": [
            {
            "span_type": "agentic.tool.invocation",
            "entities": [
                {"type": "tool", "name": "adk_book_hotel_5"},
                {"type": "agent", "name": "adk_hotel_booking_agent_5"}
            ],
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
                "expected_result": {"Precision": 0.6, "Recall": 0.6, "F1": 0.6},
                "comparer": "metric"
                }
            }
        ]
    },
]

@MonocleValidator().monocle_testcase(agent_test_cases)
async def test_run_workflows(my_test_case: TestCase):
   await MonocleValidator().test_workflow(run_agent, my_test_case)

if __name__ == "__main__":
    pytest.main([__file__]) 