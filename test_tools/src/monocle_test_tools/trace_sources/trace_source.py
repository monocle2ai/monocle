"""Trace-source abstraction.

A ``TraceSource`` represents where a test's spans come from (Okahu, a local
file, ...) and what side effects that source supports around a test run: loading
spans, setting up test cases, running filtered evals and recording a test outcome
back to the source. A provider registers one under its name (see
``monocle_test_tools.trace_sources``).
"""
from abc import ABC
from typing import Optional, Tuple


class TraceSource(ABC):
    """Base class for trace sources.

    ``record_test_result`` is a concrete no-op by default so sources that don't
    support recording an outcome (e.g. a local file source) inherit the no-op
    without extra code. Sources that do support it override the method.
    """

    #: Short name of the source, matching the ``trace_source`` string used by
    #: ``MonocleValidator`` (e.g. ``"okahu"``, ``"file"``).
    name: str = ""

    #: True if the source can select facts by a time window, which enables
    #: ``with_trace_source(..., start_time=, end_time=)`` and filtered evals.
    supports_time_window: bool = False

    #: The source's own scope name for an agentic session, if it has one, so a
    #: scope given by that name is treated as a session.
    session_scope_name: Optional[str] = None

    def setup_test_cases(self, **kwargs) -> list:
        """Build FluentTestCases from evals already recorded on this source."""
        raise NotImplementedError(f"Trace source '{self.name}' cannot set up test cases.")

    def run_filtered_eval(self, *, workflows: list, fact_name: str, start_time: str,
                          end_time: str, accepted, not_expected, eval_name: Optional[str],
                          template: Optional[dict], min_facts: int,
                          max_facts: Optional[int]) -> dict:
        """Run one eval over every fact in a time window.

        Returns the eval report (see ``monocle_test_tools.eval_report``).
        """
        raise NotImplementedError(f"Trace source '{self.name}' cannot run filtered evals.")

    def load_spans(self, *, id: str, fact_name: str, scope_name: Optional[str],
                   workflow_name: str, start_time: Optional[str] = None,
                   end_time: Optional[str] = None) -> Tuple[list, str]:
        """Fetch the spans of one fact.

        Args:
            id: The fact's id.
            fact_name: What ``id`` is: ``"trace"``, ``"session"`` or ``"scope"``.
            scope_name: The custom scope's name, when ``fact_name="scope"``.
            workflow_name: The workflow / service name the spans belong to.
            start_time / end_time: Optional window narrowing the lookup.

        Returns:
            ``(spans, source_fact_name)``: the spans, and this source's own
            name for the fact. That name is passed back to
            :meth:`record_test_result`.
        """
        raise NotImplementedError(f"Trace source '{self.name}' cannot load spans.")

    def record_test_result(self, *, fact_id: Optional[str], fact_name: Optional[str],
                           workflow_name: Optional[str], test_name: str,
                           test_failed: bool, exporters: Optional[list] = None,
                           description: str = "Test run") -> bool:
        """Record a test's outcome back to the trace source.

        Args:
            fact_id: The id of the fact the test's spans belong to (a trace id,
                session id, custom scope id, ...).
            fact_name: The source-side fact name the ``fact_id`` refers to.
            workflow_name: The workflow / service name the trace belongs to.
            test_name: The test's name; used as the recorded label.
            test_failed: Whether the test failed.
            exporters: The exporters the validator has configured, so a source
                can decide whether it is the active export target.
            description: Free-text description stored with the record.

        Returns:
            ``True`` if a record was successfully written, ``False`` otherwise
            (including the default no-op).
        """
        return False
