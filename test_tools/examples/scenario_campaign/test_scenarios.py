"""Scenario tests against the example travel agent: one live conversation per scenario.

    export OPENAI_API_KEY=...
    export MONOCLE_TEST_AGENT_MODEL=openai:gpt-4.1-mini
    pytest test_scenarios.py -v
"""
from travel_agent import build_travel_agent

from monocle_test_tools import MonocleValidator
from monocle_test_tools.runner.runner import AgentTypes


@MonocleValidator().monocle_scenario_testcase("seeds.yaml")
async def test_scenarios(scenario):
    await MonocleValidator().test_scenario_async(build_travel_agent(), AgentTypes.LANGGRAPH, scenario)
