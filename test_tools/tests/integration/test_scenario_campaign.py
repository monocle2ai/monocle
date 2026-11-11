"""A scenario is a real test of the agent's conversational behaviour.

Each entry in scenario_campaign.json is one conversation an LLM tester holds with the
travel agent in a persona, revealing withheld details only when asked. The agent passes a
withhold scenario by asking for the missing detail and then booking with it, and a
scope-escape scenario by declining the out-of-scope request. A judge that sees the agent's
reply and its tool calls decides, and the transcript is the failure message.

Needs live Azure OpenAI credentials (AZURE_OPENAI_API_DEPLOYMENT, AZURE_OPENAI_API_KEY,
AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT). Models are passed explicitly so one set of
credentials drives the target, the tester and the judge.
"""
from test_common.langgraph_travel_agent import build_chat_model, build_travel_agent

from monocle_test_tools import MonocleValidator
from monocle_test_tools.runner.runner import AgentTypes


@MonocleValidator().monocle_scenario_testcase("scenario_campaign.json")
async def test_agent_handles_each_scenario(scenario):
    result = await MonocleValidator().test_scenario_async(
        build_travel_agent(), AgentTypes.LANGGRAPH, scenario,
        model=build_chat_model(temperature=0.7), judge_model=build_chat_model())
    # test_scenario_async already asserted the pass; these say what passing meant.
    if scenario.family == "withhold":
        asked_for = {call.name for turn in result.turns for call in turn.param_tools_called}
        withheld = {p.name for p in scenario.on_request_params}
        assert withheld <= asked_for, f"the agent booked without asking for {withheld - asked_for}:\n{result.report()}"
    if scenario.family == "scope_escape":
        assert result.turns_used <= 2, f"declining should take one reply, not a negotiation:\n{result.report()}"
