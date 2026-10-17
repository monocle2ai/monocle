"""The simulated user can't leak a held-back detail: its value is not in the prompt, only behind a tool."""
from monocle_test_tools.harness.scenario import ScenarioParam, ScenarioTestCase
from monocle_test_tools.harness.test_agent import ParamCall, build_param_tools, build_system_prompt

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
