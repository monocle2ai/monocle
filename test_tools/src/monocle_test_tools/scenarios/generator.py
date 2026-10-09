"""Seeds -> scenarios, and the ``python -m monocle_test_tools scenarios`` command.

Enumeration is plain Python over the seed file and two config files
(``personas.yaml``, ``redteam.yaml``). The only model call is the seed expansion,
and only when the seed file is the minimal form.

Per seed, in this order:

    persona        the goal with every detail known, once per persona
    withhold       one detail held back until asked, once per param, then all but the first
    false_premise  a newcomer asks for a capability that does not exist
    worry          each worry the user wrote, as is
    redteam        each break mode the seed's flags expose, once per delivery

Per file: one scope_escape scenario per out_of_scope entry.
"""
import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Optional

from pydantic import ValidationError

from monocle_test_tools.harness.scenario import ScenarioParam
from monocle_test_tools.scenarios.scenario import Scenario
from monocle_test_tools.scenarios.seed import (
    MinimalSeedFile, Seed, SeedError, SeedFile, expand, load_seeds, write_seed_file,
)

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_AGENT_MODEL_ENV = "MONOCLE_TEST_AGENT_MODEL"

STUB = '''"""Scenario campaign: one live conversation per scenario in scenarios.json.

Point `build_agent` at your agent, then:   pytest {test_file} -v
"""
import pytest

from monocle_test_tools import monocle_scenarios
from monocle_test_tools.runner.runner import AgentTypes

# from my_app.agent import build_agent          # <- your agent
AGENT_TYPE = AgentTypes.LANGGRAPH                 # <- your framework


def build_agent():
    raise NotImplementedError("import and return your agent here")


@pytest.mark.asyncio
@monocle_scenarios("{scenarios_file}")
async def test_scenarios(monocle_trace_asserter, scenario):
    await scenario.run(build_agent(), AGENT_TYPE)
'''


def _config(name: str) -> dict:
    import yaml
    with open(os.path.join(CONFIG_DIR, name), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _persona(personas: dict, name: str) -> str:
    return next(p["persona"] for p in personas["personas"] if p["name"] == name)


def _params(seed: Seed, withhold: set[str]) -> list[ScenarioParam]:
    return [ScenarioParam(name=name, value=param.value, description=param.description,
                          is_initial=name not in withhold)
            for name, param in seed.params.items()]


def _make_scenario(seeds: SeedFile, seed_id: str, family: str, angle: str, *, ask: str,
                   persona: str, success: str, params: list[ScenarioParam]) -> Scenario:
    return Scenario(
        test_name=".".join(part for part in (seed_id, family, angle) if part),
        seed_id=seed_id, family=family, angle=angle, scenario=ask, persona=persona,
        params=params, success_criteria=success, max_turns=seeds.max_turns,
        target_description=seeds.agent)


def persona_scenarios(seeds: SeedFile, seed: Seed, personas: dict) -> list[Scenario]:
    return [_make_scenario(seeds, seed.id, "persona", p["name"], ask=seed.goal, persona=p["persona"],
                           success=seed.success, params=_params(seed, set()))
            for p in personas["personas"]]


def withhold_scenarios(seeds: SeedFile, seed: Seed, personas: dict) -> list[Scenario]:
    names = list(seed.params)
    if len(names) < 2:
        return []
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
    persona = personas["personas"][0]["persona"]
    return [_make_scenario(seeds, seed.id, "worry", worry.name, ask=worry.ask, persona=persona,
                           success=worry.expect, params=[])
            for worry in seed.worries]


def redteam_scenarios(seeds: SeedFile, seed: Seed, redteam: dict) -> list[Scenario]:
    out = []
    for mode_name, mode in redteam["modes"].items():
        gate = mode["gate"]
        if gate != "always" and not getattr(seed, gate):
            continue
        for delivery in mode["deliveries"]:
            out.append(_make_scenario(seeds, seed.id, "redteam", f"{mode_name}.{delivery}",
                                      ask=mode["scenario"].format(goal=seed.goal),
                                      persona=redteam["deliveries"][delivery]["persona"],
                                      success=mode["success"], params=_params(seed, set())))
    return out


def scope_escape_scenarios(seeds: SeedFile, personas: dict, redteam: dict) -> list[Scenario]:
    config = redteam["scope_escape"]
    persona = _persona(personas, config["persona"])
    return [_make_scenario(seeds, "scope", "scope_escape", _slug(topic),
                           ask=config["scenario"].format(topic=topic), persona=persona,
                           success=config["success"].format(topic=topic), params=[])
            for topic in seeds.out_of_scope]


def _slug(text: str) -> str:
    return "_".join(part for part in "".join(c if c.isalnum() else " " for c in text.lower()).split())


def generate(seeds: SeedFile, personas: Optional[dict] = None,
             redteam: Optional[dict] = None) -> list[Scenario]:
    """Every scenario the seed file supports."""
    personas = personas or _config("personas.yaml")
    redteam = redteam or _config("redteam.yaml")
    scenarios: list[Scenario] = []
    for index, seed in enumerate(seeds.seeds):
        scenarios += persona_scenarios(seeds, seed, personas)
        scenarios += withhold_scenarios(seeds, seed, personas)
        if seed.params:
            scenarios.append(false_premise_scenario(seeds, seed, index, personas, redteam))
        scenarios += worry_scenarios(seeds, seed, personas)
        scenarios += redteam_scenarios(seeds, seed, redteam)
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


def write_scenarios(scenarios: list[Scenario], path: str) -> None:
    """Write the file. ``session_id`` is left out so every run gets a fresh session."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([s.model_dump(mode="json", exclude_none=True, exclude={"session_id"})
                   for s in scenarios], handle, indent=2)


def write_stub(test_file: str, scenarios_file: str) -> bool:
    """Write the pytest stub once; never overwrite the user's edits."""
    if os.path.exists(test_file):
        return False
    with open(test_file, "w", encoding="utf-8") as handle:
        handle.write(STUB.format(test_file=os.path.basename(test_file),
                                 scenarios_file=os.path.basename(scenarios_file)))
    return True


def ask_for_seeds(path: str) -> MinimalSeedFile:
    """Three questions, when no seed file exists. Writes the answers to ``path``."""
    print("No seed file found. Three questions, then I will write it.\n")
    agent = input("What does your agent do? (one sentence)\n> ").strip()
    print("\nWhat do users come to it to do? One per line, with real details. Blank line to finish.")
    goals = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        goals.append(line)
    if not agent or not goals:
        raise SeedError("a seed needs the agent sentence and at least one goal")
    scope = input("\nAnything it should refuse? (comma separated, or blank)\n> ").strip()
    minimal = MinimalSeedFile(agent=agent, goals=goals,
                              out_of_scope=[s.strip() for s in scope.split(",") if s.strip()])
    import yaml
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(minimal.model_dump(exclude_defaults=True), handle, sort_keys=False, width=100)
    print(f"\nwrote {path}\n")
    return minimal


def _model(spec: Optional[str]) -> Any:
    from langchain.chat_models import init_chat_model
    spec = spec or os.getenv(TEST_AGENT_MODEL_ENV)
    if not spec:
        raise SeedError(f"expanding goal sentences needs a model: set {TEST_AGENT_MODEL_ENV} "
                        "(e.g. openai:gpt-4.1) or pass --model")
    return init_chat_model(spec)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m monocle_test_tools scenarios",
        description="Turn a seed file into scenarios.json and a pytest stub.")
    parser.add_argument("--seeds", default="tests/seeds.yaml", help="seed file; asked for if missing")
    parser.add_argument("--out-dir", default=None, help="where to write; default: the seed file's folder")
    parser.add_argument("--model", default=None, help=f"model for expanding goals; default ${TEST_AGENT_MODEL_ENV}")
    parser.add_argument("--no-stub", action="store_true", help="do not write test_scenarios.py")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.seeds))
    os.makedirs(out_dir, exist_ok=True)
    try:
        seeds = load_seeds(args.seeds) if os.path.isfile(args.seeds) else ask_for_seeds(args.seeds)
        if isinstance(seeds, MinimalSeedFile):
            print(f"expanding {len(seeds.goals)} goals")
            try:
                seeds = expand(seeds, _model(args.model))
            except Exception as error:  # a provider error: auth, network, quota
                raise SeedError(f"the model call to split the goals failed: {error}") from error
            expanded_path = os.path.join(out_dir, "seeds.expanded.yaml")
            write_seed_file(seeds, expanded_path)
            for seed in seeds.seeds:
                params = ", ".join(f"{n}={p.value}" for n, p in seed.params.items())
                print(f"  {seed.id:<16} params: {params}   side_effect: {'yes' if seed.side_effect else 'no'}")
            print(f"wrote {expanded_path}   (edit this if a param split is wrong)\n")
        scenarios = generate(seeds)
    except (SeedError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(summarize(scenarios))
    scenarios_path = os.path.join(out_dir, "scenarios.json")
    write_scenarios(scenarios, scenarios_path)
    print(f"\nwrote {scenarios_path}   ({len(scenarios)} scenarios; delete any you do not want)")
    if not args.no_stub:
        test_path = os.path.join(out_dir, "test_scenarios.py")
        if write_stub(test_path, scenarios_path):
            print(f"wrote {test_path}   (point build_agent at your agent, then run pytest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
