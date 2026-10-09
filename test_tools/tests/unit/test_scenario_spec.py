"""Validation of the scenario spec models. Pure Pydantic, no LLM."""
import pytest
from pydantic import ValidationError

from monocle_test_tools.harness.scenario import ScenarioParam, ScenarioTestCase

VALID = {
    "scenario": "Book a flight from source to destination.",
    "persona": "User who's foul mouthed",
    "params": [
        {"name": "source", "value": "San Francisco", "is_initial": True, "description": "source airport"},
        {"name": "destination", "value": "Seattle", "is_initial": False, "description": "destination airport"},
        {"name": "date", "value": "22nd October 2026", "is_initial": False, "description": "travel date"},
    ],
    "success_criteria": "Flight is booked",
    "max_turns": 8,
}


def test_valid_case_parses_with_a_generated_session_and_split_params():
    case = ScenarioTestCase.model_validate(VALID)
    assert case.test_name == "monocle_scenario_test" and case.max_turns == 8
    assert case.session_id.startswith("monocle_scenario_session_")
    assert [p.name for p in case.initial_params] == ["source"]
    assert [p.name for p in case.on_request_params] == ["destination", "date"]
    assert all(p.required for p in case.params)


def test_explicit_session_id_is_kept():
    assert ScenarioTestCase.model_validate({**VALID, "session_id": "my_session"}).session_id == "my_session"


def test_param_name_must_be_identifier_safe():
    with pytest.raises(ValidationError, match="identifier"):
        ScenarioParam(name="travel date", value="x", description="d")


@pytest.mark.parametrize("override, needle", [
    ({"params": [{"name": "source", "value": "SFO", "is_initial": True, "description": "a"},
                 {"name": "source", "value": "OAK", "is_initial": False, "description": "b"}]},
     "duplicate param names"),
    ({"params": [{"name": "destination", "value": "Seattle", "is_initial": False, "description": "d"}]},
     "is_initial"),
    ({"scenario": "   "}, "scenario"),
    ({"success_criteria": ""}, "success_criteria"),
    ({"max_turns": 0}, "max_turns"),
])
def test_invalid_cases_are_rejected(override, needle):
    with pytest.raises(ValidationError, match=needle):
        ScenarioTestCase.model_validate({**VALID, **override})


def test_no_params_at_all_is_allowed():
    case = ScenarioTestCase.model_validate({**VALID, "params": []})
    assert case.params == [] and case.missing_required_params(set()) == []


def test_missing_required_params_names_withheld_params_never_requested_in_order():
    case = ScenarioTestCase.model_validate(VALID)
    assert case.missing_required_params(set()) == ["destination", "date"]   # initial 'source' is exempt
    assert case.missing_required_params({"date"}) == ["destination"]
    assert case.missing_required_params({"destination", "date"}) == []


def test_missing_required_params_ignores_optional_params():
    optional = {**VALID, "params": [
        {"name": "source", "value": "SFO", "is_initial": True, "description": "a"},
        {"name": "seat", "value": "aisle", "is_initial": False, "description": "b", "required": False},
    ]}
    assert ScenarioTestCase.model_validate(optional).missing_required_params(set()) == []
