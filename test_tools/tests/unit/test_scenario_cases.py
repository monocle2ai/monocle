"""The decorator turns a seed into one pytest case per scenario: generated once and cached, or run
as written when the seed is already a list of scenarios."""
import json

import pytest

from monocle_test_tools import MonocleValidator
from monocle_test_tools.scenarios.scenario import ScenarioError, scenarios_from
from monocle_test_tools.scenarios.seed import GoalSplit, ParamSplit

ENTRY = {
    "test_name": "book_flight.withhold.date",
    "scenario": "Book a flight from one airport to another on a date",
    "persona": "terse",
    "params": [
        {"name": "source", "value": "Bombay", "description": "source airport", "is_initial": True},
        {"name": "date", "value": "22 Oct 2026", "description": "travel date"},
    ],
    "success_criteria": "The agent confirms the flight is booked.",
    "target_description": "A travel agent",
}

SEED = {"agent": "A travel agent", "goals": ["Book a flight from Bombay on 22 Oct 2026"], "out_of_scope": ["refunds"]}


class StubModel:
    """Splits every goal the same way and counts the calls; rewrites no attacks."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        if "Split it into the goal" not in prompt:
            return "no rewrite"
        self.calls += 1
        return GoalSplit(id="book_flight", goal="Book a flight on a date", success="Booked.", side_effect=False, untrusted_input=False,
                         params=[ParamSplit(name="source", value="Bombay", description="source airport"),
                                 ParamSplit(name="date", value="22 Oct 2026", description="travel date")])


def test_a_list_of_scenarios_runs_as_written(tmp_path):
    (tmp_path / "mine.json").write_text(json.dumps([ENTRY]), encoding="utf-8")
    for seed in ([ENTRY], "mine.json"):
        scenario = scenarios_from(seed, str(tmp_path))[0]
        assert scenario.test_name == "book_flight.withhold.date" and scenario.on_request_params[0].name == "date"


def test_a_seed_is_generated_once_then_read_from_the_cache_until_it_changes(tmp_path):
    model = StubModel()
    first = scenarios_from(SEED, str(tmp_path), model=model)
    again = scenarios_from(SEED, str(tmp_path), model=model)
    assert model.calls == 1 and [s.test_name for s in again] == [s.test_name for s in first]
    cache = json.loads((tmp_path / "scenarios.json").read_text())
    assert "session_id" not in cache["scenarios"][0], "every run gets a fresh session"
    scenarios_from({**SEED, "out_of_scope": ["visas"]}, str(tmp_path), model=model)
    assert model.calls == 2


def test_a_seed_without_a_model_says_how_to_configure_one(tmp_path, monkeypatch):
    monkeypatch.delenv("MONOCLE_TEST_AGENT_MODEL", raising=False)
    with pytest.raises(ValueError, match="MONOCLE_TEST_AGENT_MODEL"):
        scenarios_from(SEED, str(tmp_path))


@pytest.mark.parametrize("bad, needle", [
    ({"name": "x"}, "expected a seed"),
    ([], "non-empty"),
    ([ENTRY, ENTRY], "duplicate test_name"),
    ([{**ENTRY, "max_turns": 0}], "scenario #0 is invalid"),
    ("missing.yaml", "not found"),
])
def test_bad_seeds_are_scenario_errors(tmp_path, bad, needle):
    with pytest.raises(ScenarioError, match=needle):
        scenarios_from(bad, str(tmp_path))


def test_a_broken_yaml_file_names_the_file(tmp_path):
    (tmp_path / "seeds.yaml").write_text("agent: [unclosed\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match="seeds.yaml: not valid YAML"):
        scenarios_from("seeds.yaml", str(tmp_path))


def test_the_decorator_makes_one_case_per_scenario_named_by_test_name():
    second = {**ENTRY, "test_name": "book_flight.persona.newcomer"}

    async def test_body(scenario):
        return scenario

    marks = {m.name: m for m in MonocleValidator().monocle_scenario_testcase([ENTRY, second])(test_body).pytestmark}
    assert marks["parametrize"].kwargs["ids"] == ["book_flight.withhold.date", "book_flight.persona.newcomer"]
    assert marks["asyncio"].kwargs == {"loop_scope": "session"}, "one event loop, so shared clients stay usable"
    with pytest.raises(ValueError, match="async"):
        MonocleValidator().monocle_scenario_testcase([ENTRY])(lambda scenario: None)
