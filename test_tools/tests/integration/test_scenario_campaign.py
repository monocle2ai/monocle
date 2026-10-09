"""End-to-end: a hand-written scenario file run through the campaign runner.

Needs live Azure OpenAI credentials (AZURE_OPENAI_API_DEPLOYMENT, AZURE_OPENAI_API_KEY,
AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT), like test_scenario_harness.py. Models
are passed explicitly so one set of credentials drives the target, the tester and the
judge.
"""
import pytest
from test_common.langgraph_travel_agent import build_chat_model, build_travel_agent

from monocle_test_tools import monocle_scenarios
from monocle_test_tools.runner.runner import AgentTypes


@pytest.mark.asyncio
@monocle_scenarios("scenario_campaign.json")
async def test_scenario_campaign(monocle_trace_asserter, scenario):
    result = await scenario.run(build_travel_agent(), AgentTypes.LANGGRAPH,
                                model=build_chat_model(temperature=0.7),
                                judge_model=build_chat_model())
    assert result.turns_used >= 1, result.report()
