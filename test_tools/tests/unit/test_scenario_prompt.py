"""Tests for the test agent's system prompt. Pure string assembly, no LLM."""
from monocle_test_tools.harness.scenario import ScenarioTestCase
from monocle_test_tools.harness.test_agent import build_system_prompt

CASE = ScenarioTestCase.model_validate({
    "scenario": "Book a flight from source to destination.",
    "persona": "User who's foul mouthed",
    "params": [
        {"name": "source", "value": "San Francisco", "is_initial": True,
         "description": "source airport"},
        {"name": "destination", "value": "Seattle", "is_initial": False,
         "description": "destination airport"},
        {"name": "date", "value": "22nd October 2026", "is_initial": False,
         "description": "travel date"},
    ],
    "success_criteria": "Flight is booked",
})

TARGET_DESCRIPTION = "A travel booking agent that books flights and hotels."


def _prompt():
    return build_system_prompt(CASE, TARGET_DESCRIPTION)


def test_prompt_carries_scenario_persona_and_criteria():
    prompt = _prompt()
    assert "Book a flight from source to destination." in prompt
    assert "User who's foul mouthed" in prompt
    assert "Flight is booked" in prompt


def test_prompt_carries_the_target_description():
    assert TARGET_DESCRIPTION in _prompt()


def test_prompt_states_initial_param_values():
    assert "San Francisco" in _prompt()


def test_prompt_withholds_on_request_param_values():
    """The whole point: a withheld value must be unreachable except via its tool."""
    prompt = _prompt()
    assert "Seattle" not in prompt
    assert "22nd October 2026" not in prompt


def test_prompt_lists_on_request_param_names_and_descriptions():
    prompt = _prompt()
    assert "destination" in prompt
    assert "destination airport" in prompt
    assert "travel date" in prompt


def test_prompt_includes_the_preamble():
    from monocle_test_tools.harness.preamble import PREAMBLE
    assert PREAMBLE in _prompt()


def test_prompt_handles_a_case_with_no_params():
    case = ScenarioTestCase.model_validate({
        "scenario": "Ask for help.", "persona": "Polite user",
        "success_criteria": "Agent responds",
    })
    prompt = build_system_prompt(case, TARGET_DESCRIPTION)
    assert "WHAT YOU KNOW UP FRONT" not in prompt
    assert "DETAILS AVAILABLE ON REQUEST" not in prompt
