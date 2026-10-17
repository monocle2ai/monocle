"""A scenario, and turning a seed into a list of them.

The seed is a dict with ``agent`` and ``goals`` (or just ``agent``: the model writes the
requests and nothing is judged), or a list of scenarios to run as written, or a .yaml/.json
file holding either. Scenarios generated from a seed are cached beside
the test, keyed on a hash of the seed, so later runs make no model call.
"""
import hashlib
import json
import os
from typing import Any, Optional, Union

from pydantic import Field

from monocle_test_tools.harness.scenario import ScenarioTestCase

DEFAULT_CACHE = "scenarios.json"


class ScenarioError(ValueError):
    """An invalid seed or scenario file."""


class Scenario(ScenarioTestCase):
    family: str = Field("manual", description="persona | withhold | false_premise | redteam | scope_escape | request | manual")
    target_description: str = Field(..., description="What the target agent does, for the simulated user.")


def scenarios_from(seed: Union[dict, list, str], base_dir: str, cache: Optional[str] = None,
                   model: Any = None) -> list[Scenario]:
    data = _read(os.path.join(base_dir, seed)) if isinstance(seed, str) else seed
    if isinstance(data, list) or (isinstance(data, dict) and "scenarios" in data):
        return _validate(data, str(seed))
    if not isinstance(data, dict) or "agent" not in data:
        raise ScenarioError(f"{seed}: expected a seed (agent, and goals if you have them) or a list of scenarios")
    return _generate(data, os.path.join(base_dir, cache or DEFAULT_CACHE), model)


def _generate(seed: dict, cache_path: str, model: Any) -> list[Scenario]:
    from monocle_test_tools.harness.driver import resolve_model
    from monocle_test_tools.scenarios.generator import from_description, generate
    from monocle_test_tools.scenarios.seed import DescriptionSeed, MinimalSeedFile, expand, write_requests

    digest = hashlib.sha256(json.dumps(seed, sort_keys=True).encode()).hexdigest()[:16]
    if os.path.isfile(cache_path):
        cached = _read(cache_path)
        if isinstance(cached, dict) and cached.get("seed_hash") == digest:
            return _validate(cached, cache_path)

    model = resolve_model(model)
    if seed.get("goals"):
        split = expand(MinimalSeedFile.model_validate(seed), model)
        scenarios = generate(split, model=model)
    else:
        split = DescriptionSeed.model_validate(seed)
        scenarios = from_description(split, write_requests(split, model))
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump({"seed_hash": digest, "split_seed": split.model_dump(),
                   "scenarios": [s.model_dump(mode="json", exclude_none=True, exclude={"session_id"})
                                 for s in scenarios]}, handle, indent=2)
    return scenarios


def _read(path: str) -> Any:
    if not os.path.isfile(path):
        raise ScenarioError(f"file not found: {path}")
    kind = "YAML" if path.endswith((".yaml", ".yml")) else "JSON"
    with open(path, encoding="utf-8") as handle:
        try:
            if kind == "YAML":
                import yaml
                return yaml.safe_load(handle)
            return json.load(handle)
        except Exception as error:
            raise ScenarioError(f"{path}: not valid {kind}: {error}") from error


def _validate(data: Any, where: str) -> list[Scenario]:
    entries = data["scenarios"] if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise ScenarioError(f"{where}: expected a non-empty list of scenarios")
    scenarios = []
    for index, entry in enumerate(entries):
        try:
            scenarios.append(entry if isinstance(entry, Scenario) else Scenario.model_validate(entry))
        except Exception as error:
            raise ScenarioError(f"{where}: scenario #{index} is invalid: {error}") from error
    names = [s.test_name for s in scenarios]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ScenarioError(f"{where}: duplicate test_name {duplicates}")
    return scenarios
