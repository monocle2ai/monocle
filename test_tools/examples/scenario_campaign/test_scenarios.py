"""Scenario tests against the example travel agent: one live conversation per scenario.

    export OPENAI_API_KEY=...
    export MONOCLE_TEST_AGENT_MODEL=openai:gpt-4.1-mini
    pytest test_scenarios.py
"""
from monocle_test_tools import MonocleValidator
from travel_agent import build_travel_agent

SEED = {
    "agent": "A travel booking agent that books flights between airports and hotels in a city.",
    "goals": [
        "Book a flight from Bombay to Hyderabad on 22 Oct 2026",
        "Book a hotel in Mumbai for 27 Nov 2026",
    ],
    "out_of_scope": ["refunds", "visa advice"],
}


@MonocleValidator().monocle_scenario_testcase(SEED)
async def test_scenarios(scenario):
    await MonocleValidator().test_scenario_async(build_travel_agent(), "langgraph", scenario)
