"""Seed files: both forms, validation, and expansion with a stub model."""
import json

import pytest

from monocle_test_tools.scenarios.seed import (
    GoalSplit, MinimalSeedFile, ParamSplit, Seed, SeedError, SeedFile, SeedParam, expand,
    load_seeds, write_seed_file,
)

MINIMAL = """
agent: A travel booking agent that books flights and hotels.
goals:
  - Book a flight from Bombay to Hyderabad on 22 Oct 2026
  - Book a hotel in Mumbai for 27 Nov 2026
out_of_scope: [refunds, visa advice]
"""

EXPANDED = {
    "agent": "A travel booking agent.",
    "out_of_scope": ["refunds"],
    "seeds": [{
        "id": "book_flight",
        "goal": "Book a flight from one airport to another on a date",
        "params": {"source": {"value": "Bombay", "description": "source airport"},
                   "date": {"value": "22 Oct 2026", "description": "travel date"}},
        "success": "The agent confirms the flight is booked.",
        "side_effect": True,
    }],
}


def test_load_minimal_yaml(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    seeds = load_seeds(str(path))
    assert isinstance(seeds, MinimalSeedFile)
    assert len(seeds.goals) == 2 and seeds.out_of_scope == ["refunds", "visa advice"]


def test_load_expanded_json(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps(EXPANDED), encoding="utf-8")
    seeds = load_seeds(str(path))
    assert isinstance(seeds, SeedFile)
    assert seeds.seeds[0].params["date"].description == "travel date"
    assert seeds.seeds[0].side_effect


def test_load_reports_the_path_on_a_bad_file(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("agent: x\n", encoding="utf-8")  # no goals, no seeds
    with pytest.raises(SeedError, match="seeds.yaml"):
        load_seeds(str(path))
    with pytest.raises(SeedError, match="not found"):
        load_seeds(str(tmp_path / "missing.yaml"))


@pytest.mark.parametrize("bad, needle", [
    ({"id": "book flight", "goal": "g", "success": "s"}, "identifier-safe"),
    ({"id": "ok", "goal": "g", "success": "s", "params": {"bad name": {"value": 1, "description": "d"}}}, "tool name"),
])
def test_seed_validation_errors(bad, needle):
    with pytest.raises(Exception, match=needle):
        Seed.model_validate(bad)


def test_seed_file_rejects_duplicate_ids():
    seed = {"id": "a", "goal": "g", "success": "s"}
    with pytest.raises(Exception, match="duplicate seed ids"):
        SeedFile(agent="x", seeds=[Seed(**seed), Seed(**seed)])


class StubStructured:
    def __init__(self, splits):
        self.splits = list(splits)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.splits.pop(0)


class StubModel:
    def __init__(self, splits):
        self.structured = StubStructured(splits)

    def with_structured_output(self, schema):
        assert schema is GoalSplit
        return self.structured


def test_expand_turns_goal_sentences_into_seeds():
    minimal = MinimalSeedFile(agent="A travel agent.", goals=["Book a flight from Bombay to Hyderabad"],
                              out_of_scope=["refunds"])
    model = StubModel([GoalSplit(
        id="book_flight", goal="Book a flight from one airport to another",
        params=[ParamSplit(name="source", value="Bombay", description="source airport"),
                ParamSplit(name="destination", value="Hyderabad", description="destination airport")],
        success="The agent confirms the flight is booked.", side_effect=True)])
    seeds = expand(minimal, model)
    assert "Book a flight from Bombay to Hyderabad" in model.structured.prompts[0]
    assert seeds.out_of_scope == ["refunds"]
    seed = seeds.seeds[0]
    assert seed.id == "book_flight" and seed.side_effect
    assert seed.params["destination"].value == "Hyderabad"


def test_expand_makes_colliding_ids_unique():
    split = lambda: GoalSplit(id="book", goal="g", params=[], success="s", side_effect=False)
    seeds = expand(MinimalSeedFile(agent="a", goals=["x", "y"]), StubModel([split(), split()]))
    assert [s.id for s in seeds.seeds] == ["book", "book_2"]


def test_expand_makes_model_names_identifier_safe():
    split = GoalSplit(id="Book a Hotel!", goal="g", success="s", side_effect=True,
                      params=[ParamSplit(name="check-in date", value="27 Nov", description="d"),
                              ParamSplit(name="2nd guest", value="x", description="d")])
    seeds = expand(MinimalSeedFile(agent="a", goals=["x"]), StubModel([split]))
    assert seeds.seeds[0].id == "book_a_hotel"
    assert list(seeds.seeds[0].params) == ["check_in_date", "detail_2nd_guest"]


def test_write_then_load_round_trips(tmp_path):
    seeds = SeedFile(agent="a", seeds=[Seed(id="s", goal="g", success="ok",
                                           params={"p": SeedParam(value="v", description="d")})])
    path = str(tmp_path / "seeds.expanded.yaml")
    write_seed_file(seeds, path)
    again = load_seeds(path)
    assert isinstance(again, SeedFile)
    assert again.seeds[0].params["p"].value == "v"
