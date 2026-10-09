"""Scenario campaigns: many harness scenarios from one seed, run as pytest cases.

Install with ``pip install "monocle_test_tools[test_agent]"``; LangChain and YAML are
imported lazily, so importing ``monocle_test_tools`` needs neither.

- ``seed.py``       the seed file: the agent in a sentence, goals as sentences
- ``generator.py``  seeds -> ``scenarios.json`` (``python -m monocle_test_tools scenarios``)
- ``scenario.py``   ``Scenario``, ``load_scenarios``, ``monocle_scenarios``, ``Scenario.run``
"""
from monocle_test_tools.scenarios.scenario import (
    Scenario,
    ScenarioError,
    load_scenarios,
    monocle_scenarios,
)

__all__ = ["Scenario", "ScenarioError", "load_scenarios", "monocle_scenarios"]
