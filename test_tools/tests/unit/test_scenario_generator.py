"""One goal becomes every scenario worth running: each persona, each held-back detail, a false
premise, the red-team attacks, and one refusal per out-of-scope topic."""
from collections import Counter

from monocle_test_tools.scenarios.generator import generate
from monocle_test_tools.scenarios.seed import Seed, SeedFile, SeedParam


def flight(**overrides):
    fields = dict(id="book_flight", goal="Book a flight from one airport to another on a date",
                  success="The agent confirms the flight is booked.", side_effect=True,
                  params={"source": SeedParam(value="Bombay", description="source airport"),
                          "destination": SeedParam(value="Hyderabad", description="destination airport"),
                          "date": SeedParam(value="22 Oct 2026", description="travel date")})
    return Seed(**{**fields, **overrides})


def by_name(scenarios):
    return {s.test_name: s for s in scenarios}


def test_one_goal_yields_each_kind_of_scenario():
    scenarios = generate(SeedFile(agent="A travel agent", out_of_scope=["refunds", "visa advice"], seeds=[flight()]))
    assert Counter(s.family for s in scenarios) == {"persona": 5, "withhold": 4, "false_premise": 1,
                                                    "redteam": 6, "scope_escape": 2}
    assert len(by_name(scenarios)) == len(scenarios), "test names are unique"
    assert all(s.target_description == "A travel agent" for s in scenarios)


def test_withhold_holds_back_each_detail_then_all_but_the_first():
    scenarios = by_name(generate(SeedFile(agent="a", seeds=[flight()])))
    held = {name: [p.name for p in s.on_request_params] for name, s in scenarios.items() if ".withhold." in name}
    assert held == {"book_flight.withhold.source": ["source"], "book_flight.withhold.destination": ["destination"],
                    "book_flight.withhold.date": ["date"], "book_flight.withhold.all": ["destination", "date"]}


def test_pressure_attacks_hold_back_the_last_detail_and_only_target_goals_with_side_effects():
    scenarios = by_name(generate(SeedFile(agent="a", seeds=[flight()])))
    attack = scenarios["book_flight.redteam.unauthorised_action.urgency"]
    assert "right now" in attack.scenario and "panic" in attack.persona
    assert "travel date" in attack.success_criteria and attack.on_request_params[0].name == "date"
    read_only = generate(SeedFile(agent="a", seeds=[flight(side_effect=False)]))
    assert [s.test_name for s in read_only if s.family == "redteam"] == ["book_flight.redteam.prompt_extraction.probing"]


def test_with_a_model_attacks_are_rewritten_and_the_instruction_stays():
    class Model:
        def invoke(self, prompt):
            class Reply:
                content = ('{"input": "DRESSED", "is_valid_authority": true, "is_valid_context": true, '
                           '"is_valid_permission": true, "is_roleplay": true, "is_prompt_probing": true}')
            return Reply()

    scenarios = by_name(generate(SeedFile(agent="a", seeds=[flight()]), model=Model()))
    urgency = scenarios["book_flight.redteam.unauthorised_action.urgency"].scenario
    assert "right now" in urgency and urgency.endswith("framing:\nDRESSED")
    assert "DRESSED" not in scenarios["book_flight.redteam.unauthorised_action.plain"].scenario


def test_injection_attacks_run_only_when_the_agent_reads_outside_content():
    names = [s.test_name for s in generate(SeedFile(agent="a", seeds=[flight(untrusted_input=True)]))]
    assert {"book_flight.redteam.indirect_injection.plain", "book_flight.redteam.indirect_injection.buried"} <= set(names)
    assert not any("indirect_injection" in s.test_name for s in generate(SeedFile(agent="a", seeds=[flight()])))


def test_a_goal_with_one_detail_has_no_withhold_or_pressure_scenarios():
    one = flight(params={"city": SeedParam(value="Mumbai", description="city")})
    names = [s.test_name for s in generate(SeedFile(agent="a", seeds=[one]))]
    assert not any(".withhold." in n or "unauthorised_action" in n for n in names)


def test_each_goal_gets_a_different_false_premise():
    scenarios = generate(SeedFile(agent="a", seeds=[flight(), flight(id="book_hotel")]))
    asks = [s.scenario for s in scenarios if s.family == "false_premise"]
    assert len(asks) == 2 and asks[0] != asks[1]
