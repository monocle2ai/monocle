"""Scenario campaign against the example travel agent: one live conversation per scenario.

    export OPENAI_API_KEY=...
    pytest test_scenarios.py -v                               # red on a failing scenario
"""
import pytest
from travel_agent import build_chat_model, build_travel_agent

from monocle_test_tools import monocle_scenarios
from monocle_test_tools.runner.runner import AgentTypes


@pytest.mark.asyncio
@monocle_scenarios("scenarios.json")
async def test_scenarios(monocle_trace_asserter, scenario):
    await scenario.run(build_travel_agent(), AgentTypes.LANGGRAPH,
                       model=build_chat_model(temperature=0.7),
                       judge_model=build_chat_model())
