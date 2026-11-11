"""An older monocle_apptrace doesn't break importing test_tools; only a scenario run fails, saying what to upgrade."""
import importlib

import pytest

from monocle_apptrace.instrumentation.common import constants
from monocle_test_tools.harness import driver


def test_the_driver_imports_without_the_suppression_flag(monkeypatch):
    monkeypatch.delattr(constants, "MONOCLE_SUPPRESS_SPANS")
    try:
        assert importlib.reload(driver).MONOCLE_SUPPRESS_SPANS is None
    finally:
        monkeypatch.undo()
        importlib.reload(driver)


def test_a_scenario_run_on_an_older_apptrace_says_to_upgrade_it(monkeypatch):
    monkeypatch.setattr(driver, "MONOCLE_SUPPRESS_SPANS", None)
    harness = object.__new__(driver.ScenarioHarness)
    harness.trace_test_agent = False
    with pytest.raises(RuntimeError, match="pip install -U monocle_apptrace"):
        with harness._suppress_spans():
            pass
