"""Okahu trace source.

Loads spans from Okahu, and records a test's pass/fail outcome back to Okahu
as an evaluation label via ``POST /v1/eval/label`` (server side:
``okahu/observability/azure-fn/eval-api/function_app.py::eval_label``).
"""
import logging
import os
from typing import Optional, Tuple

import requests

from monocle_test_tools.providers.okahu.okahu_filtered_eval import OkahuFilteredEval
from monocle_test_tools.providers.okahu.okahu_span_loader import OkahuSpanLoader
from monocle_test_tools.trace_sources.trace_source import TraceSource

logger = logging.getLogger(__name__)

OKAHU_PROD_EVALUATION_ENDPOINT = "https://eval.okahu.co/api"

# Class name of the Okahu span exporter, matched by name so this module does not
# need to import the apptrace exporter package just to detect it.
OKAHU_EXPORTER_CLASS_NAME = "OkahuSpanExporter"


class OkahuTraceSource(TraceSource):
    """Trace source backed by the Okahu cloud service."""

    name = "okahu"
    supports_time_window = True

    def setup_test_cases(self, **kwargs) -> list:
        return OkahuSpanLoader.setup_test_cases(**kwargs)

    def run_filtered_eval(self, *, workflows, fact_name, start_time, end_time, accepted,
                          not_expected, eval_name, template, min_facts, max_facts) -> dict:
        return OkahuFilteredEval.from_env().run_filtered(
            workflows, accepted=accepted, not_expected=not_expected,
            eval_name=eval_name, template=template, fact_name=fact_name,
            start_time=start_time, end_time=end_time,
            min_facts=min_facts, max_facts=max_facts)

    def load_spans(self, *, id: str, fact_name: str, scope_name: Optional[str],
                   workflow_name: str, start_time: Optional[str] = None,
                   end_time: Optional[str] = None) -> Tuple[list, str]:
        window = {k: v for k, v in (("start_time", start_time), ("end_time", end_time)) if v is not None}
        if fact_name in ("session", "scope"):
            # A session is Okahu's agent_sessions scope.
            okahu_fact_name = OkahuSpanLoader.AGENT_SESSIONS_SCOPE if fact_name == "session" else scope_name
            spans = OkahuSpanLoader.load_by_scope(workflow_name=workflow_name, scope_name=okahu_fact_name,
                                                  scope_id=id, **window)
            return spans, okahu_fact_name
        spans = OkahuSpanLoader.get_spans(workflow_name=workflow_name, trace_id=id, **window)
        return spans, "traces"

    def record_test_result(self, *, fact_id: Optional[str], fact_name: Optional[str],
                           workflow_name: Optional[str], test_name: str,
                           test_failed: bool, exporters: Optional[list] = None,
                           description: str = None) -> bool:
        """Record the test outcome as an Okahu eval label.

        Best-effort: returns ``False`` (and logs at most a warning) when the
        preconditions aren't met or the request fails. Never raises, so it can't
        fail the test or mask its real result.
        """
        if not self._okahu_exporter_active(exporters):
            logger.debug("No Okahu exporter active; skipping test-result label.")
            return False

        api_key = (os.getenv("OKAHU_API_KEY") or "").strip()
        if not api_key:
            logger.debug("OKAHU_API_KEY not set; skipping test-result label.")
            return False

        if not fact_id or not workflow_name:
            logger.debug("Missing fact_id or workflow_name; skipping test-result label.")
            return False

        base = (os.getenv("OKAHU_EVALUATION_ENDPOINT") or OKAHU_PROD_EVALUATION_ENDPOINT).rstrip("/")
        value = "FAIL" if test_failed else "PASS"
        payload = {
            "result": {
                "label": test_name,
                "value": value,
                "explanation": description if description else value,
                "category": "test",
            }
        }

        try:
            response = requests.post(
                url=f"{base}/v1/eval/label",
                headers={"x-api-key": api_key},
                json=payload,
                params={
                    "trace_id": fact_id,
                    "fact_name": fact_name or "traces",
                    "workflow_name": workflow_name,
                },
                timeout=60,
            )
            response.raise_for_status()
        except Exception as exc:  # best-effort: never let recording break a test
            logger.warning("Failed to record test result label to Okahu: %s", exc)
            return False

        logger.debug("Recorded test result label for '%s' (%s) to Okahu.", test_name, value)
        return True

    @staticmethod
    def _okahu_exporter_active(exporters: Optional[list]) -> bool:
        """Return True if an Okahu span exporter is present in ``exporters``."""
        if not exporters:
            return False
        return any(type(e).__name__ == OKAHU_EXPORTER_CLASS_NAME for e in exporters)
