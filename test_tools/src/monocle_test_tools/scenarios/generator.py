"""A split seed -> scenarios, from two config files: personas.yaml and redteam.yaml.

Per goal: one per persona, one per held-back detail plus one holding back all but the
first, a false premise, and the red-team attacks. Per seed: one per out-of-scope topic.
"""
import os
from typing import Any

import yaml

from monocle_test_tools.harness.scenario import ScenarioParam
from monocle_test_tools.scenarios.attacks import dress
from monocle_test_tools.scenarios.scenario import Scenario
from monocle_test_tools.scenarios.seed import DescriptionSeed, Seed, SeedFile, identifier, unique

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def config(name: str) -> dict:
    with open(os.path.join(CONFIG_DIR, name), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate(seeds: SeedFile, model: Any = None) -> list[Scenario]:
    """Every scenario for the seed. ``model`` rewrites the red-team attacks; without one
    they stay as written."""
    personas = {p["name"]: p["persona"] for p in config("personas.yaml")["personas"]}
    redteam = config("redteam.yaml")

    def scenario(name, ask, persona, success, params=(), family="", min_turns=1):
        return Scenario(test_name=name, family=family, scenario=ask, persona=persona, params=list(params),
                        success_criteria=success, max_turns=seeds.max_turns,
                        min_turns=min(min_turns, seeds.max_turns), target_description=seeds.agent)

    out = []
    for index, seed in enumerate(seeds.seeds):
        for name, persona in personas.items():
            out.append(scenario(f"{seed.id}.persona.{name}", seed.goal, persona, seed.success,
                                _params(seed), "persona"))

        names = list(seed.params)
        if len(names) >= 2:     # one detail must be stated up front to open with
            all_label = unique("all", set(names))       # in case a detail is itself called "all"
            for held, label in [({n}, n) for n in names] + [(set(names[1:]), all_label)]:
                out.append(scenario(f"{seed.id}.withhold.{label}", seed.goal, personas["terse"],
                                    seed.success, _params(seed, held), "withhold"))

        if seed.params:
            premise = redteam["false_premise"]
            fake = premise["asks"][index % len(premise["asks"])]
            out.append(scenario(f"{seed.id}.false_premise", premise["scenario"].format(goal=seed.goal, fake=fake),
                                personas["newcomer"], f"{seed.success} {premise['success'].format(fake=fake)}",
                                _params(seed), "false_premise"))

        out += _redteam(seed, redteam, model, scenario)

    used: set[str] = set()
    escape = redteam["scope_escape"]
    for topic in seeds.out_of_scope:
        out.append(scenario(f"scope.scope_escape.{unique(identifier(topic, 'topic'), used)}",
                            escape["scenario"].format(topic=topic), personas["power_user"],
                            escape["success"].format(topic=topic), family="scope_escape"))
    return out


def _redteam(seed: Seed, redteam: dict, model: Any, scenario) -> list[Scenario]:
    out = []
    for mode_name, mode in redteam["modes"].items():
        if mode.get("needs") and not getattr(seed, mode["needs"]):
            continue
        held, missing = set(), "a detail"
        if mode.get("withhold_last_detail"):
            if len(seed.params) < 2:
                continue        # pressure to act needs a detail to act without
            last = list(seed.params)[-1]
            held, missing = {last}, seed.params[last].description
        known = [p for name, p in seed.params.items() if name not in held]
        baseline = mode["scenario"].format(goal=seed.goal, missing=missing)
        if known:               # the user's real details, so a rewrite doesn't invent its own
            baseline += " The user's details: " + ", ".join(f"{p.description} {p.value}" for p in known) + "."
        for delivery in mode["deliveries"]:
            dressed = dress(model, delivery, baseline)
            ask = baseline if not dressed else (
                f"{baseline}\n\nOpen with this message, as written, then keep pushing in its framing:\n{dressed}")
            # The held-back detail is optional: the agent passes by asking for it, and the
            # user may never get to answer, so the judge decides rather than the tool log.
            out.append(scenario(f"{seed.id}.redteam.{mode_name}.{delivery}", ask,
                                redteam["deliveries"][delivery], mode["success"].format(missing=missing),
                                _params(seed, held, optional=held), "redteam", mode.get("min_turns", 1)))
    return out


def _params(seed: Seed, held: frozenset = frozenset(), optional: frozenset = frozenset()) -> list[ScenarioParam]:
    return [ScenarioParam(name=name, value=p.value, description=p.description,
                          is_initial=name not in held, required=name not in optional)
            for name, p in seed.params.items()]


def from_description(seed: DescriptionSeed, requests: list[str]) -> list[Scenario]:
    """One unjudged conversation per request, each played by the next persona."""
    personas = config("personas.yaml")["personas"]
    return [Scenario(test_name=f"request_{i + 1:02d}.{personas[i % len(personas)]['name']}", family="request",
                     scenario=request, persona=personas[i % len(personas)]["persona"],
                     max_turns=seed.max_turns, target_description=seed.agent)
            for i, request in enumerate(requests)]
