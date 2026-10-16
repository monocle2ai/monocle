"""Scenario tests: many harness scenarios from one seed, run as pytest cases.

Use them through ``MonocleValidator().monocle_scenario_testcase(seed)`` and
``MonocleValidator().test_scenario_async(agent, agent_type, scenario)``.

- ``seed.py``       the seed: the agent in a sentence, goals as sentences, what it refuses
- ``generator.py``  an expanded seed -> scenarios, across personas, withheld details and attacks
- ``attacks.py``    red-team text rewritten with DeepTeam's vendored templates
- ``scenario.py``   ``Scenario``, and turning a seed, an expanded seed or a list into scenarios
"""
from monocle_test_tools.scenarios.scenario import Scenario, ScenarioError, load_scenarios

__all__ = ["Scenario", "ScenarioError", "load_scenarios"]
