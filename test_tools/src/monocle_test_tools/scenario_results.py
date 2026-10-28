"""Opt-in record of scenario runs under ``.monocle/test_results/<run>/``, next to ``test_traces``.

Each run gets ``summary.txt`` (one line per scenario, numbered in the order they ran) and one
``<number>_<test name>.txt`` per failed scenario with the full report. The report carries the
trace id of every turn, so a failure can be looked up in the traces locally or in Okahu.

Off by default; enabled by ``pytest --monocle-scenario-results`` or ``MONOCLE_SCENARIO_RESULTS=true``.
"""
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from monocle_apptrace.exporters.file_exporter import DEFAULT_TRACE_FOLDER

if TYPE_CHECKING:
    from monocle_test_tools.harness.result import ScenarioResult

RESULTS_ENV = "MONOCLE_SCENARIO_RESULTS"

_enabled = False
_started: Optional[datetime] = None
_run_dir: Optional[str] = None
_results: list["ScenarioResult"] = []


def enable() -> None:
    global _enabled
    _enabled = True


def is_enabled() -> bool:
    return _enabled or os.getenv(RESULTS_ENV, "").lower() in ("1", "true")


def run_dir() -> Optional[str]:
    """Where this run's results are, or None when nothing has been recorded."""
    return _run_dir


def record(result: "ScenarioResult") -> Optional[str]:
    """Add ``result`` to this run. Returns the path of its report when it failed, else None."""
    global _started, _run_dir
    if not is_enabled():
        return None
    if _run_dir is None:
        _started = datetime.now()
        _run_dir = os.path.join(".", DEFAULT_TRACE_FOLDER, "test_results", _started.strftime("%Y-%m-%d_%H.%M.%S"))
        os.makedirs(_run_dir, exist_ok=True)
    _results.append(result)

    path = None
    if not result.passed:
        name = re.sub(r"[^\w.-]", "_", result.test_name)
        path = os.path.join(_run_dir, f"{len(_results):02d}_{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(result.report() + "\n\n  scenario\n    " + result.scenario.replace("\n", "\n    ") + "\n")
    _write_summary()
    return path


def _write_summary() -> None:
    failed = sum(not r.passed for r in _results)
    width = max(len(r.test_name) for r in _results)
    lines = [f"Scenario results  {_started:%Y-%m-%d %H:%M:%S}",
             f"{len(_results) - failed} passed, {failed} failed", "",
             f"   #  result  turns  {'test':<{width}}  reason"]
    for number, r in enumerate(_results, 1):
        reason = "" if r.passed else r.headline.replace("\n", " ")
        lines.append(f"  {number:>2}  {'passed' if r.passed else 'FAILED'}  "
                     f"{f'{r.turns_used}/{r.max_turns}':<5}  {r.test_name:<{width}}  {reason}".rstrip())
    with open(os.path.join(_run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
