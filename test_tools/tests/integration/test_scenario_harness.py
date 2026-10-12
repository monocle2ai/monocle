"""End-to-end test of the scenario harness against a live LangGraph travel agent.

Needs live Azure OpenAI credentials: AZURE_OPENAI_API_DEPLOYMENT, AZURE_OPENAI_API_KEY,
AZURE_OPENAI_API_VERSION and AZURE_OPENAI_ENDPOINT (the repo's test.env supplies them).

The test agent and judge models are passed explicitly rather than resolved from
MONOCLE_TEST_AGENT_MODEL, so one set of Azure credentials drives the target agent, the
test agent and the judge. The env-var route stays available for providers that
init_chat_model can build without extra arguments.
"""
import json
import os

import pytest
from test_common.langgraph_travel_agent import build_chat_model, build_travel_agent

from monocle_test_tools.harness import ScenarioHarness, ScenarioTestCase
from monocle_test_tools.runner.runner import AgentTypes

TARGET_DESCRIPTION = (
    "A travel booking agent that books a flight from one airport to another for a "
    "given date, and a hotel in a given city for a given date.")

# Span assertions stay in the test body rather than the scenario JSON (spec non-goal).
EXPECTED_TOOL = {
    "book_flight_foul_mouthed_user": "book_flight",
    "book_hotel_terse_user": "book_hotel",
}


def _scenarios():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "scenario_test_cases.json"), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _scenarios(), ids=lambda case: case["test_name"])
async def test_scenario_against_travel_agent(case, monocle_trace_asserter):
    harness = ScenarioHarness(ScenarioTestCase.model_validate(case),
                              model=build_chat_model(temperature=0.7),
                              judge_model=build_chat_model())

    result = await harness.run_scenario_async(
        build_travel_agent(), AgentTypes.LANGGRAPH, TARGET_DESCRIPTION)

    assert result.passed, result.report()
    # The booking tool ran, and only target spans are in the pool -- the test agent's
    # own LLM calls are suppressed at the wrapper.
    monocle_trace_asserter.called_tool(EXPECTED_TOOL[case["test_name"]])


@pytest.mark.asyncio
async def test_withheld_params_are_only_revealed_on_request(monocle_trace_asserter):
    """The test agent must ask for the destination rather than volunteering it."""
    case = ScenarioTestCase.model_validate(_scenarios()[0])
    harness = ScenarioHarness(case, model=build_chat_model(temperature=0.7),
                              judge_model=build_chat_model())

    result = await harness.run_scenario_async(
        build_travel_agent(), AgentTypes.LANGGRAPH, TARGET_DESCRIPTION)

    assert result.passed, result.report()
    opening = result.turns[0].tester_message
    assert "San Francisco" in opening, (
        "the opening message must carry the initial param to the target agent:\n"
        + result.report())
    assert result.turns[0].param_tools_called == [], (
        "the opening message must not require a withheld param:\n" + result.report())
    revealed = [call.name for turn in result.turns
                for call in turn.param_tools_called]
    assert "destination" in revealed, (
        "the destination should have been fetched via its tool:\n" + result.report())
