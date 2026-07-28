"""Trace-source abstraction.

A ``TraceSource`` represents where a test's spans come from (Okahu, a local
file, ...) and what side effects that source supports around a test run. Today
the only behavior modeled here is recording a test outcome back to the source
(see :meth:`TraceSource.record_test_result`); the existing span loaders
(``OkahuSpanLoader``, ``JSONSpanLoader``) are expected to migrate under this
abstraction in the future, so the interface is intentionally kept general.
"""
from abc import ABC
from typing import Optional


class TraceSource(ABC):
    """Base class for trace sources.

    ``record_test_result`` is a concrete no-op by default so sources that don't
    support recording an outcome (e.g. a local file source) inherit the no-op
    without extra code. Sources that do support it override the method.
    """

    #: Short name of the source, matching the ``trace_source`` string used by
    #: ``MonocleValidator`` (e.g. ``"okahu"``, ``"file"``).
    name: str = ""

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
