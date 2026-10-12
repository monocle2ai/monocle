"""Validation tests for the scenario spec models. Pure Pydantic, no LLM."""
import pytest
from pydantic import ValidationError

from monocle_test_tools.harness.scenario import ScenarioParam, ScenarioTestCase

VALID = {
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
    "max_turns": 8,
}


def test_valid_case_parses():
    case = ScenarioTestCase.model_validate(VALID)
    assert case.test_name == "monocle_scenario_test"
    assert case.max_turns == 8
    assert len(case.params) == 3


def test_session_id_is_auto_generated():
    case = ScenarioTestCase.model_validate(VALID)
    assert case.session_id.startswith("monocle_scenario_session_")


def test_explicit_session_id_is_kept():
    case = ScenarioTestCase.model_validate({**VALID, "session_id": "my_session"})
    assert case.session_id == "my_session"


def test_params_split_into_initial_and_on_request():
    case = ScenarioTestCase.model_validate(VALID)
    assert [p.name for p in case.initial_params] == ["source"]
    assert [p.name for p in case.on_request_params] == ["destination", "date"]


def test_param_name_must_be_identifier_safe():
    with pytest.raises(ValidationError, match="identifier"):
        ScenarioParam(name="travel date", value="x", description="d")


def test_duplicate_param_names_rejected():
    dupes = {**VALID, "params": [
        {"name": "source", "value": "SFO", "is_initial": True, "description": "a"},
        {"name": "source", "value": "OAK", "is_initial": False, "description": "b"},
    ]}
    with pytest.raises(ValidationError, match="duplicate param names"):
        ScenarioTestCase.model_validate(dupes)


def test_params_without_any_initial_rejected():
    none_initial = {**VALID, "params": [
        {"name": "destination", "value": "Seattle", "is_initial": False,
         "description": "destination airport"},
    ]}
    with pytest.raises(ValidationError, match="is_initial"):
        ScenarioTestCase.model_validate(none_initial)


def test_no_params_at_all_is_allowed():
    case = ScenarioTestCase.model_validate({**VALID, "params": []})
    assert case.params == []


def test_empty_scenario_rejected():
    with pytest.raises(ValidationError, match="scenario"):
        ScenarioTestCase.model_validate({**VALID, "scenario": "   "})


def test_empty_success_criteria_rejected():
    with pytest.raises(ValidationError, match="success_criteria"):
        ScenarioTestCase.model_validate({**VALID, "success_criteria": ""})


def test_max_turns_below_one_rejected():
    with pytest.raises(ValidationError, match="max_turns"):
        ScenarioTestCase.model_validate({**VALID, "max_turns": 0})


# --- required params -------------------------------------------------------

def test_required_defaults_to_true():
    case = ScenarioTestCase.model_validate(VALID)
    assert all(p.required for p in case.params)


def test_required_can_be_turned_off():
    param = ScenarioParam(name="seat", value="aisle", description="seat preference",
                          required=False)
    assert param.required is False


def test_missing_required_params_reports_unrequested_on_request_params():
    case = ScenarioTestCase.model_validate(VALID)
    assert case.missing_required_params(set()) == ["destination", "date"]


def test_missing_required_params_exempts_initial_params():
    """An initial param is satisfied by construction -- it went out up front."""
    case = ScenarioTestCase.model_validate(VALID)
    assert "source" not in case.missing_required_params(set())


def test_missing_required_params_is_empty_when_all_were_requested():
    case = ScenarioTestCase.model_validate(VALID)
    assert case.missing_required_params({"destination", "date"}) == []


def test_missing_required_params_ignores_optional_params():
    optional = {**VALID, "params": [
        {"name": "source", "value": "SFO", "is_initial": True, "description": "a"},
        {"name": "seat", "value": "aisle", "is_initial": False, "description": "b",
         "required": False},
    ]}
    case = ScenarioTestCase.model_validate(optional)
    assert case.missing_required_params(set()) == []


def test_missing_required_params_preserves_declaration_order():
    case = ScenarioTestCase.model_validate(VALID)
    assert case.missing_required_params({"date"}) == ["destination"]


def test_missing_required_params_with_no_params_at_all():
    case = ScenarioTestCase.model_validate({**VALID, "params": []})
    assert case.missing_required_params(set()) == []
