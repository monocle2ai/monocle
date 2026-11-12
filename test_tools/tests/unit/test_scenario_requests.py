"""A seed with only the agent's description: the model writes realistic requests, each runs as a
conversation for the full turn limit, and nothing is judged, so it passes unless the agent errors."""
import pytest
from langchain_core.messages import AIMessage

from monocle_test_tools.harness import driver
from monocle_test_tools.harness.scenario import ScenarioTestCase
from monocle_test_tools.harness.test_agent import build_system_prompt
from monocle_test_tools.scenarios.scenario import scenarios_from
from monocle_test_tools.scenarios.seed import Requests

DESCRIPTION = {"agent": "A travel booking agent that books flights and hotels."}


class Writer:
    def __init__(self, n):
        self.n, self.prompts = n, []

    def with_structured_output(self, schema):
        assert schema is Requests
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return Requests(requests=[f"Book trip number {i}." for i in range(self.n)])


def test_a_description_becomes_unjudged_conversations_of_eight_turns(tmp_path):
    writer = Writer(60)
    scenarios = scenarios_from(DESCRIPTION, str(tmp_path), model=writer)
    assert "Write 50 different requests" in writer.prompts[0] and len(scenarios) == 50
    assert scenarios[0].test_name == "request_01.power_user" and scenarios[1].test_name == "request_02.newcomer"
    assert all(s.success_criteria is None and s.max_turns == 8 and not s.params for s in scenarios)
    scenarios_from(DESCRIPTION, str(tmp_path), model=writer)
    assert len(writer.prompts) == 1, "the second run reads the cache"


def test_count_and_max_turns_come_from_the_seed(tmp_path):
    scenarios = scenarios_from({**DESCRIPTION, "count": 3, "max_turns": 5}, str(tmp_path), model=Writer(3))
    assert len(scenarios) == 3 and scenarios[0].max_turns == 5


def test_the_simulated_user_gets_no_criteria_just_keeps_going():
    prompt = build_system_prompt(ScenarioTestCase(scenario="Book a hotel in Pune.", persona="terse"), "A travel agent")
    assert "[SUCCESS CRITERIA]" not in prompt and "There is no checklist" in prompt


@pytest.mark.asyncio
async def test_an_unjudged_conversation_runs_every_turn_and_passes_without_a_judge(monkeypatch):
    class Tester:
        async def ainvoke(self, state):
            return {"messages": [*state["messages"], AIMessage(content="A hotel in Pune, please.")]}

    class Runner:
        async def end_session(self, session_id):
            pass

    class Validator:
        spans, _test_all_up_spans, calls = (), (), 0

        async def run_agent_async(self, *args, **kwargs):
            Validator.calls += 1
            return "Which dates?"

    monkeypatch.setattr(driver, "build_test_agent", lambda case, description, model: (Tester(), []))
    monkeypatch.setattr(driver, "get_agent_runner", lambda agent_type: Runner())
    monkeypatch.setattr(driver, "ResponseEvaluator", lambda *a: pytest.fail("a judge was built"))
    harness = driver.ScenarioHarness(ScenarioTestCase(scenario="Book a hotel.", persona="terse", max_turns=8),
                                     model=object(), trace_test_agent=True)
    harness.validator = Validator()
    result = await harness.run_scenario_async(object(), "langgraph", "A travel agent")
    assert result.passed and result.failure_reason is None and Validator.calls == 8
