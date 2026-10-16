"""A seed, an expanded seed or a list of scenarios becomes pytest cases, one per conversation."""
import json

import pytest

from monocle_test_tools import MonocleValidator
from monocle_test_tools.scenarios import scenario as scenario_module
from monocle_test_tools.scenarios.scenario import Scenario, ScenarioError, load_scenarios, scenarios_from
from monocle_test_tools.scenarios.seed import GoalSplit, ParamSplit

ENTRY = {
    "test_name": "book_flight.withhold.date",
    "scenario": "Book a flight from one airport to another on a date",
    "persona": "terse",
    "params": [
        {"name": "source", "value": "Bombay", "description": "source airport", "is_initial": True},
        {"name": "date", "value": "22 Oct 2026", "description": "travel date", "is_initial": False},
    ],
    "success_criteria": "The agent confirms the flight is booked.",
    "target_description": "A travel agent",
}

EXPANDED = {
    "agent": "A travel agent",
    "out_of_scope": ["refunds"],
    "seeds": [{"id": "book_flight", "goal": "Book a flight", "success": "Booked.",
               "params": {"source": {"value": "Bombay", "description": "source airport"},
                          "date": {"value": "22 Oct 2026", "description": "travel date"}}}],
}

SEED = {"agent": "A travel agent", "goals": ["Book a flight from Bombay on 22 Oct 2026"], "out_of_scope": ["refunds"]}


def write(tmp_path, data, name="scenarios.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path.name


class StubStructured:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return GoalSplit(id="book_flight", goal="Book a flight from one airport on a date", success="Booked.",
                         side_effect=False,
                         params=[ParamSplit(name="source", value="Bombay", description="source airport"),
                                 ParamSplit(name="date", value="22 Oct 2026", description="travel date")])


class StubModel:
    def __init__(self):
        self.structured = StubStructured()

    def with_structured_output(self, schema):
        return self.structured

    def invoke(self, prompt):          # the attack rewriter: no JSON, so attacks stay plain
        return "no rewrite"


def test_a_list_of_scenarios_runs_as_is(tmp_path):
    scenarios = scenarios_from([ENTRY], str(tmp_path))
    assert [s.test_name for s in scenarios] == ["book_flight.withhold.date"]
    assert isinstance(scenarios[0], Scenario) and scenarios[0].on_request_params[0].name == "date"
    assert scenarios[0].session_id


def test_a_scenario_file_is_read_relative_to_the_test(tmp_path):
    name = write(tmp_path, [ENTRY], "mine.json")
    assert scenarios_from(name, str(tmp_path))[0].test_name == "book_flight.withhold.date"


def test_an_expanded_seed_needs_no_model_and_writes_the_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("MONOCLE_TEST_AGENT_MODEL", raising=False)
    scenarios = scenarios_from(EXPANDED, str(tmp_path))
    assert any(s.family == "withhold" for s in scenarios)
    cache = json.loads((tmp_path / "scenarios.json").read_text())
    assert cache["seed_hash"] and len(cache["scenarios"]) == len(scenarios)
    assert "session_id" not in cache["scenarios"][0], "every run gets a fresh session"


def test_a_seed_is_expanded_once_then_served_from_the_cache(tmp_path):
    model = StubModel()
    first = scenarios_from(SEED, str(tmp_path), model=model)
    again = scenarios_from(SEED, str(tmp_path), model=model)
    assert model.structured.calls == 1, "the second run must not call the model"
    assert [s.test_name for s in again] == [s.test_name for s in first]
    assert json.loads((tmp_path / "scenarios.json").read_text())["expanded_seed"]["seeds"][0]["id"] == "book_flight"


def test_editing_the_seed_regenerates(tmp_path):
    model = StubModel()
    scenarios_from(SEED, str(tmp_path), model=model)
    scenarios_from({**SEED, "out_of_scope": ["refunds", "visas"]}, str(tmp_path), model=model)
    assert model.structured.calls == 2


def test_a_seed_without_a_model_says_how_to_configure_one(tmp_path, monkeypatch):
    monkeypatch.delenv("MONOCLE_TEST_AGENT_MODEL", raising=False)
    with pytest.raises(ValueError, match="MONOCLE_TEST_AGENT_MODEL"):
        scenarios_from(SEED, str(tmp_path))


def test_the_cache_path_can_be_chosen(tmp_path):
    scenarios_from(EXPANDED, str(tmp_path), cache="flight.scenarios.json")
    assert (tmp_path / "flight.scenarios.json").is_file()


@pytest.mark.parametrize("bad, needle", [
    ({"agent": "x"}, "expected a seed"),
    ([], "non-empty"),
    ([ENTRY, ENTRY], "duplicate test_name"),
    ([{**ENTRY, "max_turns": 0}], "scenario #0 is invalid"),
    ([{k: v for k, v in ENTRY.items() if k != "target_description"}], "target_description"),
])
def test_bad_sources_are_scenario_errors(tmp_path, bad, needle):
    with pytest.raises(ScenarioError, match=needle):
        scenarios_from(bad, str(tmp_path))


def test_broken_or_missing_files_are_scenario_errors(tmp_path):
    (tmp_path / "broken.json").write_text("[{", encoding="utf-8")
    with pytest.raises(ScenarioError, match="not valid JSON"):
        load_scenarios(str(tmp_path / "broken.json"))
    with pytest.raises(ScenarioError, match="not found"):
        scenarios_from("missing.yaml", str(tmp_path))


def test_the_decorator_parametrizes_one_case_per_scenario_named_by_test_name(tmp_path, monkeypatch):
    second = {**ENTRY, "test_name": "book_flight.persona.newcomer"}
    monkeypatch.setattr(scenario_module, "DEFAULT_CACHE", "unused.json")
    decorate = MonocleValidator().monocle_scenario_testcase([ENTRY, second])

    async def test_body(scenario):
        return scenario

    wrapped = decorate(test_body)
    marks = {m.name: m for m in wrapped.pytestmark}
    assert marks["parametrize"].kwargs["ids"] == ["book_flight.withhold.date", "book_flight.persona.newcomer"]
    assert "asyncio" in marks


def test_the_decorator_refuses_a_sync_test():
    with pytest.raises(ValueError, match="async"):
        MonocleValidator().monocle_scenario_testcase([ENTRY])(lambda scenario: None)
