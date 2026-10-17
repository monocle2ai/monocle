"""A scenario rejects what would make a run meaningless, and knows which held-back details the agent never asked for."""
import pytest
from pydantic import ValidationError

from monocle_test_tools.harness.scenario import ScenarioParam, ScenarioTestCase

VALID = {
    "scenario": "Book a flight from source to destination.",
    "persona": "An impatient user",
    "params": [
        {"name": "source", "value": "San Francisco", "is_initial": True, "description": "source airport"},
        {"name": "destination", "value": "Seattle", "description": "destination airport"},
        {"name": "seat", "value": "aisle", "description": "seat", "required": False},
    ],
    "success_criteria": "Flight is booked",
}


def test_a_valid_scenario_gets_a_session_and_splits_its_details():
    case = ScenarioTestCase.model_validate(VALID)
    assert case.session_id.startswith("monocle_scenario_session_")
    assert [p.name for p in case.initial_params] == ["source"]
    assert [p.name for p in case.on_request_params] == ["destination", "seat"]


@pytest.mark.parametrize("override, needle", [
    ({"params": [{"name": "a", "value": 1, "is_initial": True, "description": "d"},
                 {"name": "a", "value": 2, "description": "d"}]}, "duplicate param names"),
    ({"params": [{"name": "a", "value": 1, "description": "d"}]}, "is_initial"),
    ({"success_criteria": ""}, "success_criteria"),
    ({"max_turns": 0}, "max_turns"),
    ({"min_turns": 11}, "min_turns"),
])
def test_invalid_scenarios_are_rejected(override, needle):
    with pytest.raises(ValidationError, match=needle):
        ScenarioTestCase.model_validate({**VALID, **override})


def test_a_param_name_must_work_as_a_tool_name():
    with pytest.raises(ValidationError, match="identifier"):
        ScenarioParam(name="travel date", value="x", description="d")


def test_missing_required_params_skips_stated_and_optional_details():
    case = ScenarioTestCase.model_validate(VALID)
    assert case.missing_required_params(set()) == ["destination"]
    assert case.missing_required_params({"destination"}) == []
