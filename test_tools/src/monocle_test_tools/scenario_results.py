"""Opt-in log of scenario runs at ``.monocle/test_results/scenarios_<time>.log``, next to ``test_traces``.

One line per passed scenario and the full report per failed one, written as each finishes. The
report carries each turn's trace id, to look the failure up in the traces locally or in Okahu.
Off by default; ``pytest --monocle-scenario-results`` or ``MONOCLE_SCENARIO_RESULTS=true`` turns it on.
"""
import os
import textwrap
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from monocle_apptrace.exporters.file_exporter import DEFAULT_TRACE_FOLDER

if TYPE_CHECKING:
    from monocle_test_tools.harness.result import ScenarioResult

RESULTS_ENV = "MONOCLE_SCENARIO_RESULTS"

_enabled = False
_path: Optional[str] = None


def pytest_addoption(parser) -> None:
    parser.addoption("--monocle-scenario-results", action="store_true", default=False,
                     help="Log scenario results to .monocle/test_results/scenarios_<time>.log. "
                          f"Also enabled by {RESULTS_ENV}=true.")


def pytest_configure(config) -> None:
    global _enabled
    _enabled = config.getoption("monocle_scenario_results", default=False)


def pytest_terminal_summary(terminalreporter) -> None:
    if _path:
        terminalreporter.write_line(f"Scenario results: {_path}")


def is_enabled() -> bool:
    return _enabled or os.getenv(RESULTS_ENV, "").lower() in ("1", "true")


def reset() -> None:
    """Forget the current log. Exposed for test isolation."""
    global _enabled, _path
    _enabled, _path = False, None


def record(result: "ScenarioResult") -> Optional[str]:
    """Append ``result`` to this run's log and return the log's path, or None when logging is off."""
    global _path
    if not is_enabled():
        return None
    if _path is None:
        folder = os.path.join(".", DEFAULT_TRACE_FOLDER, "test_results")
        os.makedirs(folder, exist_ok=True)
        _path = os.path.join(folder, f"scenarios_{datetime.now():%Y-%m-%d_%H.%M.%S}.log")
    if result.passed:
        entry = f"PASSED  {result.test_name}\n"
    else:
        entry = f"FAILED  {result.test_name}\n{textwrap.indent(result.report(), '  ')}\n\n"
    with open(_path, "a", encoding="utf-8") as f:
        f.write(entry)
    return _path
