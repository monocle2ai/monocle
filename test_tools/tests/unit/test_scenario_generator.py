"""The generator: which scenarios a seed yields, gated on what the seed says."""
import json

import pytest

from monocle_test_tools.scenarios.generator import generate, summarize, write_scenarios, write_stub
from monocle_test_tools.scenarios.scenario import load_scenarios
from monocle_test_tools.scenarios.seed import Seed, SeedError, SeedFile, SeedParam, Worry


def flight(**overrides):
    fields = dict(id="book_flight", goal="Book a flight from one airport to another on a date",
                  success="The agent confirms the flight is booked.", side_effect=True,
                  params={"source": SeedParam(value="Bombay", description="source airport"),
                          "destination": SeedParam(value="Hyderabad", description="destination airport"),
                          "date": SeedParam(value="22 Oct 2026", description="travel date")})
    fields.update(overrides)
    return Seed(**fields)


def families(scenarios, seed_id="book_flight"):
    out = {}
    for s in scenarios:
        if s.seed_id == seed_id:
            out[s.family] = out.get(s.family, 0) + 1
    return out


def test_one_seed_yields_every_family_it_supports():
    scenarios = generate(SeedFile(agent="A travel agent", out_of_scope=["refunds"], seeds=[flight()]))
    assert families(scenarios) == {"persona": 5, "withhold": 4, "false_premise": 1, "redteam": 4}
    assert families(scenarios, "scope") == {"scope_escape": 1}
    assert len({s.test_name for s in scenarios}) == len(scenarios)


def test_withhold_variants_hold_back_one_param_then_all_but_the_first():
    scenarios = generate(SeedFile(agent="a", seeds=[flight()]))
    withhold = {s.angle: [p.name for p in s.on_request_params] for s in scenarios if s.family == "withhold"}
    assert withhold == {"source": ["source"], "destination": ["destination"], "date": ["date"],
                        "all": ["destination", "date"]}
    for s in scenarios:
        assert s.initial_params or not s.params, "the harness needs at least one initial param"


def test_persona_variants_know_everything():
    scenarios = generate(SeedFile(agent="a", seeds=[flight()]))
    personas = [s for s in scenarios if s.family == "persona"]
    assert len(personas) == 5 and all(not s.on_request_params for s in personas)
    assert len({s.persona for s in personas}) == 5


def test_redteam_modes_are_gated_on_the_seed_flags():
    safe = generate(SeedFile(agent="a", seeds=[flight(side_effect=False)]))
    assert {s.angle for s in safe if s.family == "redteam"} == {"prompt_extraction.plain"}
    risky = generate(SeedFile(agent="a", seeds=[flight(side_effect=True, untrusted_input=True)]))
    angles = {s.angle for s in risky if s.family == "redteam"}
    assert "unauthorised_action.authority" in angles and "indirect_injection.buried" in angles


def test_redteam_scenarios_carry_the_attack_and_the_delivery_persona():
    scenarios = generate(SeedFile(agent="a", seeds=[flight()]))
    attack = next(s for s in scenarios if s.angle == "unauthorised_action.urgency")
    assert "immediately" in attack.scenario and "panic" in attack.persona
    assert "confirms the details" in attack.success_criteria


def test_scope_escape_one_per_topic():
    scenarios = generate(SeedFile(agent="a", out_of_scope=["refunds", "visa advice"], seeds=[flight()]))
    scope = [s for s in scenarios if s.family == "scope_escape"]
    assert [s.angle for s in scope] == ["refunds", "visa_advice"]
    assert "visa advice" in scope[1].scenario and "visa advice" in scope[1].success_criteria
    assert not scope[0].params


def test_worries_become_scenarios_as_written():
    worry = Worry(name="past_date", ask="Book a hotel in Mumbai for 1 Jan 2020",
                  expect="The agent points out the date is in the past and does not book.")
    scenarios = generate(SeedFile(agent="a", seeds=[flight(worries=[worry])]))
    w = next(s for s in scenarios if s.family == "worry")
    assert w.test_name == "book_flight.worry.past_date" and w.scenario == worry.ask
    assert w.success_criteria == worry.expect and not w.params


def test_colliding_scenario_names_are_refused_with_the_cause():
    clash = flight(params={"all": SeedParam(value="x", description="d"),
                           "date": SeedParam(value="y", description="d")})
    with pytest.raises(SeedError, match="book_flight.withhold.all"):
        generate(SeedFile(agent="a", seeds=[clash]))


def test_each_seed_gets_a_different_false_premise_ask():
    scenarios = generate(SeedFile(agent="a", seeds=[flight(), flight(id="book_hotel")]))
    asks = [s.scenario for s in scenarios if s.family == "false_premise"]
    assert len(asks) == 2 and asks[0] != asks[1]


def test_a_seed_with_one_param_has_no_withhold_variants():
    one = flight(params={"city": SeedParam(value="Mumbai", description="city")})
    assert "withhold" not in families(generate(SeedFile(agent="a", seeds=[one])))


def test_every_scenario_carries_the_agent_description_and_turn_budget():
    scenarios = generate(SeedFile(agent="A travel agent", max_turns=6, seeds=[flight()]))
    assert all(s.target_description == "A travel agent" and s.max_turns == 6 for s in scenarios)


def test_summarize_counts_per_seed_and_family():
    text = summarize(generate(SeedFile(agent="a", out_of_scope=["refunds"], seeds=[flight()])))
    assert "book_flight" in text and "5  persona" in text and "scope_escape" in text


def test_written_scenarios_load_back_with_a_fresh_session(tmp_path):
    scenarios = generate(SeedFile(agent="a", seeds=[flight()]))
    path = str(tmp_path / "scenarios.json")
    write_scenarios(scenarios, path)
    again = load_scenarios(path)
    assert [s.test_name for s in again] == [s.test_name for s in scenarios]
    assert "session_id" not in json.load(open(path, encoding="utf-8"))[0]
    assert again[0].session_id != scenarios[0].session_id


def test_write_stub_once_and_never_overwrite(tmp_path):
    test_file = str(tmp_path / "test_scenarios.py")
    assert write_stub(test_file, str(tmp_path / "scenarios.json"))
    with open(test_file, encoding="utf-8") as handle:
        text = handle.read()
    assert 'monocle_scenarios("scenarios.json")' in text and "build_agent" in text
    with open(test_file, "w", encoding="utf-8") as handle:
        handle.write("edited")
    assert not write_stub(test_file, str(tmp_path / "scenarios.json"))
    with open(test_file, encoding="utf-8") as handle:
        assert handle.read() == "edited"
