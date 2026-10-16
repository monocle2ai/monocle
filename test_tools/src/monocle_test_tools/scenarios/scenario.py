"""A scenario, and turning whatever the user wrote into a list of them.

``MonocleValidator().monocle_scenario_testcase(source)`` calls ``scenarios_from`` at
collection time. ``source`` is one of:

    a seed           {"agent": ..., "goals": [...], "out_of_scope": [...]}, or a .yaml/.json file of one
    an expanded seed {"agent": ..., "seeds": [...]}: goals already split into details
    scenarios        a list of scenario dicts, or a .json file holding one, run as is

A seed is expanded and enumerated once and cached beside the test, keyed on a hash of the
seed, so later runs make no model call and the generated scenarios can be read and committed.
"""
import hashlib
import json
import os
from typing import Any, Optional, Union

from pydantic import Field

from monocle_test_tools.harness.scenario import ScenarioTestCase

DEFAULT_CACHE = "scenarios.json"


class ScenarioError(ValueError):
    """Raised for an invalid scenario source; the message carries the path and the case."""


class Scenario(ScenarioTestCase):
    """The harness case, plus where it came from."""

    seed_id: str = Field("manual", description="Seed this scenario was generated from.")
    family: str = Field("manual", description="persona | withhold | false_premise | scope_escape | redteam | worry | manual")
    angle: str = Field("", description="The variant within the family, e.g. 'date' or 'authority'.")
    target_description: str = Field(..., description="What the target agent does, for the tester's prompt.")


def scenarios_from(source: Union[dict, list, str], base_dir: str, cache: Optional[str] = None,
                   model: Any = None) -> list[Scenario]:
    """The scenarios ``source`` describes; see the module docstring for the accepted shapes."""
    data = _read(os.path.join(base_dir, source)) if isinstance(source, str) else source
    if isinstance(data, list) or (isinstance(data, dict) and "scenarios" in data):
        return _validate(data, str(source))
    if not isinstance(data, dict) or not ({"goals", "seeds"} & set(data)):
        raise ScenarioError(f"{source}: expected a seed (agent + goals), an expanded seed "
                            "(agent + seeds) or a list of scenarios")
    return _from_seed(data, os.path.join(base_dir, cache or DEFAULT_CACHE), model)


def _from_seed(seed: dict, cache_path: str, model: Any) -> list[Scenario]:
    """Generate from a seed, or reuse the cached result when the seed has not changed."""
    from monocle_test_tools.harness.driver import resolve_model
    from monocle_test_tools.scenarios.generator import generate
    from monocle_test_tools.scenarios.seed import MinimalSeedFile, SeedFile, expand

    digest = hashlib.sha256(json.dumps(seed, sort_keys=True).encode()).hexdigest()[:16]
    if os.path.isfile(cache_path):
        cached = _read(cache_path)
        if isinstance(cached, dict) and cached.get("seed_hash") == digest:
            return _validate(cached, cache_path)

    if "goals" in seed:
        model = resolve_model(model)       # splitting goal sentences needs a model
        expanded = expand(MinimalSeedFile.model_validate(seed), model)
    else:
        expanded = SeedFile.model_validate(seed)
        model = model or _optional_model()  # dressing attacks is optional
    scenarios = generate(expanded, model=model)
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump({"seed_hash": digest,
                   "expanded_seed": expanded.model_dump(exclude_none=True),
                   "scenarios": [s.model_dump(mode="json", exclude_none=True, exclude={"session_id"})
                                 for s in scenarios]}, handle, indent=2)
    return scenarios


def _optional_model() -> Any:
    from monocle_test_tools.harness.driver import TEST_AGENT_MODEL_ENV, resolve_model
    return resolve_model(None) if os.getenv(TEST_AGENT_MODEL_ENV) else None


def load_scenarios(path: str) -> list[Scenario]:
    """Load a scenario file: a JSON list of scenarios, or a cache file written by a seed."""
    return _validate(_read(path), path)


def _read(path: str) -> Any:
    if not os.path.isfile(path):
        raise ScenarioError(f"file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    try:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(text)
        return json.loads(text)
    except Exception as error:
        raise ScenarioError(f"{path}: not valid {'YAML' if path.endswith(('.yaml', '.yml')) else 'JSON'}: {error}") from error


def _validate(data: Any, where: str) -> list[Scenario]:
    entries = data["scenarios"] if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise ScenarioError(f"{where}: expected a non-empty list of scenarios")
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        try:
            scenario = entry if isinstance(entry, Scenario) else Scenario.model_validate(entry)
        except Exception as error:
            raise ScenarioError(f"{where}: scenario #{index} is invalid: {error}") from error
        if scenario.test_name in seen:
            raise ScenarioError(f"{where}: duplicate test_name '{scenario.test_name}'")
        seen.add(scenario.test_name)
        scenarios.append(scenario)
    return scenarios
