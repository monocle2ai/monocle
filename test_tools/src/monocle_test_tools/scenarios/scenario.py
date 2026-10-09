"""A scenario: one harness case plus where it came from, loaded from a file.

Modelled on ``csv_cases.py``: a case type, a loader with its own error, a ``run``
method on the case, and a decorator that parametrizes a test over the file.

    @monocle_scenarios("scenarios.json")
    async def test_scenarios(monocle_trace_asserter, scenario):
        await scenario.run(build_agent(), AgentTypes.LANGGRAPH)
"""
import inspect
import json
import os
from typing import Any

import pytest
from pydantic import Field

from monocle_test_tools.harness.result import ScenarioResult
from monocle_test_tools.harness.scenario import ScenarioTestCase


class ScenarioError(ValueError):
    """Raised for an invalid scenario file; the message carries the path and the case."""


class Scenario(ScenarioTestCase):
    """The harness case, plus where it came from."""

    seed_id: str = Field("manual", description="Seed this scenario was generated from.")
    family: str = Field("manual", description="persona | withhold | false_premise | scope_escape | redteam | worry | manual")
    angle: str = Field("", description="The variant within the family, e.g. 'date' or 'authority'.")
    target_description: str = Field(..., description="What the target agent does, for the tester's prompt.")

    async def run(self, agent: Any, agent_type: str, *, model: Any = None,
                  judge_model: Any = None) -> ScenarioResult:
        """Run the conversation and assert it passed, with the transcript as the message."""
        from monocle_test_tools.harness.driver import ScenarioHarness

        harness = ScenarioHarness(self, model=model, judge_model=judge_model)
        result = await harness.run_scenario_async(agent, agent_type, self.target_description)
        assert result.passed, result.report()
        return result


def load_scenarios(path: str) -> list[Scenario]:
    """Load a JSON array of scenarios. Validates each and rejects duplicate names."""
    if not os.path.isfile(path):
        raise ScenarioError(f"scenario file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list) or not entries:
        raise ScenarioError(f"{path}: expected a non-empty JSON array of scenarios")

    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        try:
            scenario = Scenario.model_validate(entry)
        except Exception as error:
            raise ScenarioError(f"{path}: scenario #{index} is invalid: {error}") from error
        if scenario.test_name in seen:
            raise ScenarioError(f"{path}: duplicate test_name '{scenario.test_name}'")
        seen.add(scenario.test_name)
        scenarios.append(scenario)
    return scenarios


def monocle_scenarios(path: str):
    """Parametrize a test over a scenario file, one pytest case per scenario.

    Relative paths resolve against the calling test file's directory, so the test
    works regardless of pytest's invocation directory.
    """
    resolved = path
    if not os.path.isabs(path):
        caller_file = inspect.stack()[1].filename
        resolved = os.path.join(os.path.dirname(os.path.abspath(caller_file)), path)
    scenarios = load_scenarios(resolved)
    return pytest.mark.parametrize("scenario", scenarios, ids=[s.test_name for s in scenarios])
