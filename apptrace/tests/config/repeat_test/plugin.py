from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pytest
from _pytest.config import Config
from _pytest.runner import TestReport


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("monocle pass rate")
    group.addoption(
        "--repeat-count",
        dest="repeat_count",
        action="store",
        type=int,
        default=1,
        help="Number of times to repeat each test (default: 1)",
    )
    group.addoption(
        "--min-pass-rate",
        dest="min_pass_rate",
        action="store",
        type=float,
        default=1.0,
        help="Minimum pass rate (0.0-1.0) required for tests (default: 1.0)",
    )


def pytest_configure(config: Config) -> None:
    repeat_count = config.getoption("repeat_count")
    min_pass_rate = config.getoption("min_pass_rate")

    if not 0.0 <= min_pass_rate <= 1.0:
        raise pytest.UsageError("--min-pass-rate must be between 0 and 1")

    if repeat_count > 1:
        tracker = PassRateTracker(
            repeat_count=repeat_count,
            min_pass_rate=min_pass_rate,
            config=config,
        )
        config.pluginmanager.register(tracker, name="monocle-pass-rate-tracker")


@dataclass
class RunResult:
    outcome: str
    duration: float
    error: str | None = None


@dataclass
class TestStats:
    passes: int = 0
    failures: int = 0
    skipped: int = 0
    runs: List[RunResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passes + self.failures

    @property
    def pass_ratio(self) -> float:
        if self.total == 0:
            return 1.0
        return self.passes / self.total


class PassRateTracker:
    """
    Pytest plugin that tracks pass rates across repeated test runs.

    This plugin repeats each test multiple times and tracks the pass/fail ratio.
    If the pass rate for any test falls below the configured threshold, the test
    suite is marked as failed.

    Attributes:
        repeat_count: Number of times to repeat each test
        min_pass_rate: Minimum required pass rate (0.0-1.0) for tests to succeed
        config: Pytest configuration object
        results: Dictionary mapping test node IDs to their statistics
        failed_tests: List of tests that failed to meet the minimum pass rate
    """

    def __init__(self, repeat_count: int, min_pass_rate: float, config: Config) -> None:
        """
        initialize the PassRateTracker.

        Args:
            repeat_count: Number of times to repeat each test
            min_pass_rate: Minimum pass rate (0.0-1.0) required for tests
            config: Pytest configuration object
        """
        self.repeat_count = repeat_count
        self.min_pass_rate = min_pass_rate
        self.config = config
        self.results: Dict[str, TestStats] = {}
        self.failed_tests: List[tuple[str, TestStats, float]] = []

    def pytest_runtestloop(self, session) -> bool:
        """
        Override the default test execution loop to repeat tests.

        This hook replaces pytest's standard test loop with one that runs
        each test item multiple times according to repeat_count.

        Args:
            session: The pytest session object containing collected test items

        Returns:
            True to indicate the test loop has been handled
        """
        items = session.items
        count = len(items)

        for idx, item in enumerate(items):
            nextitem = items[idx + 1] if idx + 1 < count else None
            for _ in range(self.repeat_count):
                session.config.hook.pytest_runtest_protocol(
                    item=item,
                    nextitem=nextitem,
                )

        return True

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        """
        Process test reports and collect pass/fail statistics.

        This hook is called for each test phase (setup, call, teardown).
        Only the "call" phase is tracked for pass rate calculations.

        Args:
            report: Test report containing outcome and error information
        """
        if report.when != "call":
            return

        nodeid = report.nodeid
        stats = self.results.setdefault(nodeid, TestStats())

        run = RunResult(
            outcome=report.outcome,
            duration=getattr(report, "duration", 0.0),
        )

        if report.passed:
            stats.passes += 1
        elif report.failed:
            stats.failures += 1
            run.error = (
                report.longreprtext
                if hasattr(report, "longreprtext")
                else str(report.longrepr)
            )
        elif report.skipped:
            stats.skipped += 1

        stats.runs.append(run)

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        """
        Finalize pass rate calculations and set exit status.

        Called after all tests have completed. Checks each test's pass rate
        against the minimum threshold and sets the session exit status to
        TESTS_FAILED if any test is below the threshold.

        Args:
            session: The pytest session object
            exitstatus: The current exit status code
        """
        for nodeid, stats in self.results.items():
            ratio = stats.pass_ratio
            if ratio < self.min_pass_rate:
                self.failed_tests.append((nodeid, stats, ratio))

        if self.failed_tests and exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter) -> None:
        """
        Display test pass rate summary in the terminal.

        Shows a summary of test results with pass rates. The level of detail
        depends on verbosity:
        - verbosity 0: No summary
        - verbosity 1 (-v): Pass rates for each test
        - verbosity 2 (-vv): Pass rates + individual run details + error previews

        Args:
            terminalreporter: Pytest terminal reporter for output
        """
        if not self.results:
            return

        verbosity = terminalreporter.verbosity

        if verbosity < 1:
            return

        terminalreporter.section("Test Run Summary", sep="=")

        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].pass_ratio,
        )

        for nodeid, stats in sorted_results:
            if stats.total == 0:
                continue

            ratio = stats.pass_ratio
            passed = ratio >= self.min_pass_rate

            status = "✓ PASS" if passed else "✗ FAIL"
            color = "green" if passed else "red"

            terminalreporter.write_line(
                f"{status} {nodeid}: {stats.passes}/{stats.total} ({ratio:.1%})",
                **{color: True},
            )

            if verbosity >= 2:
                for idx, run in enumerate(stats.runs, 1):
                    outcome_color = (
                        "green" if run.outcome == "passed"
                        else "red" if run.outcome == "failed"
                        else "yellow"
                    )

                    terminalreporter.write_line(
                        f"    Run {idx}: {run.outcome.upper()} ({run.duration:.2f}s)",
                        **{outcome_color: True},
                    )

                    if run.outcome == "failed" and run.error and verbosity >= 2:
                        for line in run.error.splitlines()[:3]:
                            terminalreporter.write_line(
                                f"      {line}",
                                red=True,
                            )

        if self.failed_tests:
            terminalreporter.section("Tests below pass rate threshold", sep="-")
            for nodeid, stats, ratio in self.failed_tests:
                terminalreporter.write_line(
                    f"{nodeid}: {stats.passes}/{stats.total} "
                    f"({ratio:.1%}) < {self.min_pass_rate:.1%}",
                    red=True,
                )

