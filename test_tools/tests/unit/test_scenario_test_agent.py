"""The simulated user can't leak a held-back detail: its value is not in the prompt, only behind a tool.
It only ever plays the customer: the target agent's replies reach it marked as the target agent's."""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from monocle_test_tools.harness import driver
from monocle_test_tools.harness.evaluator import Verdict
from monocle_test_tools.harness.scenario import ScenarioParam, ScenarioTestCase
from monocle_test_tools.harness.test_agent import TARGET_REPLIED, ParamCall, build_param_tools, build_system_prompt

CASE = ScenarioTestCase.model_validate({
    "scenario": "Book a flight from source to destination.",
    "persona": "An impatient user",
    "params": [{"name": "source", "value": "San Francisco", "is_initial": True, "description": "source airport"},
               {"name": "destination", "value": "Seattle", "description": "destination airport"},
               {"name": "nights", "value": 4, "description": "number of nights"}],
    "success_criteria": "Flight is booked",
})


def test_the_prompt_states_known_details_and_names_held_back_ones_without_their_values():
    prompt = build_system_prompt(CASE, "A travel agent.")
    for text in ("A travel agent.", CASE.scenario, "An impatient user", "Flight is booked", "San Francisco",
                 "destination airport", "number of nights"):
        assert text in prompt
    assert "Seattle" not in prompt


def test_each_tool_returns_its_own_value_and_records_the_call():
    tools, calls = build_param_tools(CASE.params)
    assert [t.name for t in tools] == ["source", "destination", "nights"]
    assert [t.invoke({}) for t in tools[1:]] == ["Seattle", "4"]
    assert calls == [ParamCall(name="destination", value="Seattle"), ParamCall(name="nights", value="4")]


def test_the_prompt_keeps_the_simulated_user_in_the_customers_role():
    prompt = build_system_prompt(CASE, "A travel agent.")
    assert "Never take the target agent's role" in prompt
    assert TARGET_REPLIED.strip() not in prompt   # naming the marker makes the model write the agent's lines


class _Tester:
    """The simulated user: records the messages it is given and says the same thing every turn."""
    def __init__(self):
        self.received = []

    async def ainvoke(self, state):
        self.received.append(list(state["messages"]))
        return {"messages": list(state["messages"]) + [AIMessage(content="I want to book a flight.")]}


class _Judge:
    def __init__(self, model, success_criteria):
        pass

    async def judge(self, response, transcript, spans=()):
        return Verdict(met=False, reason="keep going")


class _Runner:
    async def end_session(self, session_id):
        return None


def test_the_target_agents_reply_reaches_the_simulated_user_marked_as_the_target_agent(monkeypatch):
    tester = _Tester()

    async def agent_turn(*args, **kwargs):
        return "How can I assist you today?"

    monkeypatch.setattr(driver, "resolve_model", lambda spec, *a, **k: object())
    monkeypatch.setattr(driver, "build_test_agent", lambda case, description, model: (tester, []))
    monkeypatch.setattr(driver, "get_agent_runner", lambda agent_type: _Runner())
    monkeypatch.setattr(driver, "ResponseEvaluator", _Judge)
    harness = driver.ScenarioHarness(CASE.model_copy(update={"max_turns": 2}), model=object())
    monkeypatch.setattr(harness.validator, "run_agent_async", agent_turn)
    asyncio.run(harness.run_scenario_async(object(), "langgraph", "A travel agent."))

    reply = tester.received[1][-1]
    assert isinstance(reply, HumanMessage)
    assert reply.content == "The target agent replied:\nHow can I assist you today?"
