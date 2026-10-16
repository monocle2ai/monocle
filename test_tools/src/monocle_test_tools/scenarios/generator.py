"""Seeds -> scenarios.

Enumeration is plain Python over the seed file and two config files
(``personas.yaml``, ``redteam.yaml``). The model is used for two things only: splitting
goal sentences into details when the seed is the minimal form, and rewriting the
red-team attacks into DeepTeam's framings. Both fall back cleanly without one.

Per seed, in this order:

    persona        the goal with every detail known, once per persona
    withhold       one detail held back until asked, once per param, then all but the first
    false_premise  a newcomer asks for a capability that does not exist
    worry          each worry the user wrote, as is
    redteam        each break mode the seed's flags expose, once per delivery

Per file: one scope_escape scenario per out_of_scope entry.
"""
import os
from collections import Counter
from typing import Any, Optional

from monocle_test_tools.harness.scenario import ScenarioParam
from monocle_test_tools.scenarios.attacks import TEMPLATES, dress
from monocle_test_tools.scenarios.scenario import Scenario
from monocle_test_tools.scenarios.seed import Seed, SeedError, SeedFile

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def _config(name: str) -> dict:
    import yaml
    with open(os.path.join(CONFIG_DIR, name), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _persona(personas: dict, name: str) -> str:
    for preset in personas["personas"]:
        if preset["name"] == name:
            return preset["persona"]
    raise SeedError(f"personas.yaml names no persona '{name}'")


def _params(seed: Seed, withhold: set[str], optional: set[str] = frozenset()) -> list[ScenarioParam]:
    return [ScenarioParam(name=name, value=param.value, description=param.description,
                          is_initial=name not in withhold, required=name not in optional)
            for name, param in seed.params.items()]


def _make_scenario(seeds: SeedFile, seed_id: str, family: str, angle: str, *, ask: str,
                   persona: str, success: str, params: list[ScenarioParam],
                   min_turns: int = 1) -> Scenario:
    return Scenario(
        test_name=".".join(part for part in (seed_id, family, angle) if part),
        seed_id=seed_id, family=family, angle=angle, scenario=ask, persona=persona,
        params=params, success_criteria=success, max_turns=seeds.max_turns,
        min_turns=min(min_turns, seeds.max_turns), target_description=seeds.agent)


def persona_scenarios(seeds: SeedFile, seed: Seed, personas: dict) -> list[Scenario]:
    return [_make_scenario(seeds, seed.id, "persona", p["name"], ask=seed.goal, persona=p["persona"],
                           success=seed.success, params=_params(seed, set()))
            for p in personas["personas"]]


def withhold_scenarios(seeds: SeedFile, seed: Seed, personas: dict) -> list[Scenario]:
    names = list(seed.params)
    if len(names) < 2:
        return []          # the harness needs one detail stated up front and one to withhold
    persona = _persona(personas, personas["withholding"])
    variants = [({name}, name) for name in names] + [(set(names[1:]), "all")]
    return [_make_scenario(seeds, seed.id, "withhold", angle, ask=seed.goal, persona=persona,
                           success=seed.success, params=_params(seed, withhold))
            for withhold, angle in variants]


def false_premise_scenario(seeds: SeedFile, seed: Seed, index: int, personas: dict,
                           redteam: dict) -> Scenario:
    config = redteam["false_premise"]
    fake = config["asks"][index % len(config["asks"])]      # a different fake ask per seed
    return _make_scenario(seeds, seed.id, "false_premise", "",
                          ask=config["scenario"].format(goal=seed.goal, fake=fake),
                          persona=_persona(personas, config["persona"]),
                          success=f"{seed.success} {config['success_suffix'].format(fake=fake)}",
                          params=_params(seed, set()))


def worry_scenarios(seeds: SeedFile, seed: Seed, personas: dict) -> list[Scenario]:
    persona = _persona(personas, personas["default"])
    return [_make_scenario(seeds, seed.id, "worry", worry.name, ask=worry.ask, persona=persona,
                           success=worry.expect, params=[])
            for worry in seed.worries]


def redteam_scenarios(seeds: SeedFile, seed: Seed, redteam: dict, model: Any = None,
                      stats: Optional[Counter] = None) -> list[Scenario]:
    """One scenario per (mode, delivery). With a model, DeepTeam's template rewrites the
    attack into the delivery's framing; ``stats`` counts how many were dressed or kept plain."""
    stats = stats if stats is not None else Counter()
    out = []
    for mode_name, mode in redteam["modes"].items():
        gate = mode["gate"]
        if gate != "always" and not getattr(seed, gate):
            continue
        # A pressure attack needs something to skip: the last detail is withheld, so
        # acting anyway means inventing it. Without two details there is nothing to withhold.
        if mode.get("withhold_last_detail"):
            if len(seed.params) < 2:
                continue
            withhold = {list(seed.params)[-1]}
            missing = seed.params[next(iter(withhold))].description
        else:
            withhold, missing = set(), "a detail"
        baseline = mode["scenario"].format(goal=seed.goal, missing=missing) + _details(seed, withhold)
        # The withheld detail is optional: the agent passes by asking, and the tester may
        # never get to answer, so the judge decides rather than the tool log.
        params = _params(seed, withhold, optional=withhold)
        for delivery in mode["deliveries"]:
            dressed = dress(model, delivery, baseline)
            if delivery in TEMPLATES:          # 'plain' and 'buried' have no template; they are not fallbacks
                stats["dressed" if dressed else "plain"] += 1
            out.append(_make_scenario(seeds, seed.id, "redteam", f"{mode_name}.{delivery}",
                                      ask=_attack(baseline, dressed),
                                      persona=redteam["deliveries"][delivery]["persona"],
                                      success=mode["success"].format(missing=missing),
                                      params=params, min_turns=mode.get("min_turns", 1)))
    return out


def _attack(baseline: str, dressed: Optional[str]) -> str:
    """The tester's instruction: the baseline, plus DeepTeam's rewrite as the opening message."""
    if not dressed:
        return baseline
    return f"{baseline}\n\nOpen with this message, as written, then keep pushing in its framing:\n{dressed}"


def _details(seed: Seed, withhold: set[str]) -> str:
    """The user's real details (minus any withheld), so a rewritten attack does not invent its own."""
    known = [p for name, p in seed.params.items() if name not in withhold]
    if not known:
        return ""
    return " The user's details: " + ", ".join(f"{p.description} {p.value}" for p in known) + "."


def scope_escape_scenarios(seeds: SeedFile, personas: dict, redteam: dict) -> list[Scenario]:
    config = redteam["scope_escape"]
    persona = _persona(personas, config["persona"])
    return [_make_scenario(seeds, "scope", "scope_escape", _slug(topic),
                           ask=config["scenario"].format(topic=topic), persona=persona,
                           success=config["success"].format(topic=topic), params=[])
            for topic in seeds.out_of_scope]


def _slug(text: str) -> str:
    return "_".join(part for part in "".join(c if c.isalnum() else " " for c in text.lower()).split())


def generate(seeds: SeedFile, personas: Optional[dict] = None, redteam: Optional[dict] = None,
             model: Any = None, stats: Optional[Counter] = None) -> list[Scenario]:
    """Every scenario the seed file supports. ``model`` dresses the red-team attacks."""
    personas = personas or _config("personas.yaml")
    redteam = redteam or _config("redteam.yaml")
    scenarios: list[Scenario] = []
    for index, seed in enumerate(seeds.seeds):
        scenarios += persona_scenarios(seeds, seed, personas)
        scenarios += withhold_scenarios(seeds, seed, personas)
        if seed.params:    # the fake ask is "also do X with it"; it needs a real request to hang on
            scenarios.append(false_premise_scenario(seeds, seed, index, personas, redteam))
        scenarios += worry_scenarios(seeds, seed, personas)
        scenarios += redteam_scenarios(seeds, seed, redteam, model, stats)
    scenarios += scope_escape_scenarios(seeds, personas, redteam)

    duplicates = sorted(name for name, n in Counter(s.test_name for s in scenarios).items() if n > 1)
    if duplicates:
        raise SeedError(f"these scenario names would collide: {duplicates}. Rename the param, "
                        "worry or out-of-scope entry behind each one.")
    return scenarios


def summarize(scenarios: list[Scenario]) -> str:
    """Per seed, how many scenarios of each family."""
    lines = []
    for seed_id in dict.fromkeys(s.seed_id for s in scenarios):
        mine = [s for s in scenarios if s.seed_id == seed_id]
        lines.append(f"  {seed_id:<16}-> {len(mine)} scenarios")
        for family, n in Counter(s.family for s in mine).items():
            lines.append(f"      {n:>3}  {family}")
    return "\n".join(lines)
