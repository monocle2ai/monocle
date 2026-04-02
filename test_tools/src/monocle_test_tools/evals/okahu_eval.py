import json
import logging
import os
from datetime import datetime, timezone
from opentelemetry.sdk.trace import Span
import requests
from monocle_apptrace.exporters.okahu import okahu_exporter
from monocle_apptrace.exporters.okahu.okahu_eval_result_exporter import OkahuEvalResultExporter
from monocle_test_tools.evals.base_eval import BaseEval
from typing import Optional, Union

logger = logging.getLogger(__name__)
OKAHU_PROD_EVALUATION_ENDPOINT = "https://eval.okahu.co/api"

class OkahuEval(BaseEval):
    def __init__(self, **data):
        eval_options = data.get("eval_options")
        super().__init__(eval_options=eval_options)
        # Instance-level state tracking for hybrid approach
        self._trace_exported = False
        self._current_trace_id = None
    
    @staticmethod
    def _map_fact_name(fact_name: str) -> str:
        """
        Map user-friendly fact names to internal Okahu fact names.
        
        User-facing terms:
        - "traces" -> "traces"
        - "inferences" -> "inferences"
        - "agentic_turns" -> "agent_requests"
        - "agentic_sessions" -> "agent_sessions"
        - "agent_invocation" -> "agent_operations"
        - "tool_execution" -> "tool_operations"
        - "commits" -> "git_commits"
        
        Args:
            fact_name: The user-provided fact name
            
        Returns:
            The internal fact name used by Okahu
        """
        mapping = {
            "traces": "traces",
            "inferences": "inferences",
            "agentic_turns": "agent_requests",
            "agentic_sessions": "agent_sessions",
            "agent_invocation": "agent_operations",
            "tool_execution": "tool_operations",
            "commits": "git_commits",
            "conversations": "conversations",
            "test_runs": "test_runs",
            "tests": "tests"
        }
        
        mapped = mapping.get(fact_name)
        if mapped is None:
            raise ValueError(
                f"Invalid fact_name '{fact_name}'. Supported values: {', '.join(sorted(set(mapping.keys())))}"
            )
        return mapped
    
    def export_trace(self, filtered_spans: list[Span]) -> str:
        """Export trace to Okahu evaluation service once per test."""
        if not filtered_spans:
            raise ValueError("No spans to export for evaluation")
        
        span = filtered_spans[0]
        trace_id = format(span.get_span_context().trace_id, '032x')
        
        # Skip if already exported
        if self._trace_exported and self._current_trace_id == trace_id:
            return trace_id
        
        # Store for later cleanup
        self._current_trace_id = trace_id
        
        # Get API credentials
        api_key = (os.getenv("OKAHU_API_KEY")).strip()
        if not api_key:
            raise AssertionError("OKAHU_API_KEY is not configured.")
        
        # Export spans to Okahu
        exporter = okahu_exporter.OkahuSpanExporter(evaluate=True)
        exporter.export(filtered_spans)
        exporter.shutdown()
        
        self._trace_exported = True
        return trace_id
    
    def verify_eval_template_exists(self, eval_name: str, fact_name: str = "traces"):
        """Helper method to verify the specified evaluation template exists in Okahu before submitting eval job.
        
        Note: fact_name should already be the internal name (already mapped) when this method is called.
        """
        api_key = (os.getenv("OKAHU_API_KEY")).strip()
        if not api_key:
            raise AssertionError("OKAHU_API_KEY is not configured.")
        
        base = os.getenv("OKAHU_EVALUATION_ENDPOINT", OKAHU_PROD_EVALUATION_ENDPOINT).rstrip("/")
        list_url = f"{base}/v1/eval/templates"
        headers = {"x-api-key": api_key}
        params = {"fact_name": fact_name}
        
        try:
            response = requests.get(url=list_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            templates = response.json().get("templates", [])
            
            # Check if a template exists with both matching name and group_by
            matching_template = None
            for template in templates:
                if template.get("name") == eval_name and template.get("group_by") == fact_name:
                    matching_template = template
                    break
            
            if not matching_template:
                # Format available templates to show only name and group_by
                available_templates = [
                    {"name": t.get("name"), "group_by": t.get("group_by")} 
                    for t in templates
                ]
                raise AssertionError(
                    f"Evaluation template with name '{eval_name}' and group_by '{fact_name}' not found in Okahu. "
                    f"Available templates: {available_templates}"
                )
        except requests.RequestException as exc:
            raise AssertionError(f"Failed to verify evaluation template existence: {exc}") from exc

    def enumerate_fact_ids(self, filtered_spans: list[Span], fact_name: str) -> list[str]:
        """Enumerate unique fact IDs for a supported fact name from spans."""
        if not filtered_spans:
            return []

        def _trace_id(span: Span) -> str:
            return format(span.get_span_context().trace_id, '032x')

        def _span_id(span: Span) -> str:
            return format(span.get_span_context().span_id, '016x')

        def _attr(span: Span, key: str) -> str:
            value = span.attributes.get(key)
            if not value:
                return ""
            str_value = str(value).strip()
            # Strip 0x prefix if present (for hex IDs)
            if str_value.startswith("0x"):
                str_value = str_value[2:]
            return str_value

        def _span_type(span: Span) -> str:
            return _attr(span, "span.type")

        unique_ids: set[str] = set()
        ordered_ids: list[str] = []

        def _add(candidate: str) -> None:
            if candidate and candidate not in unique_ids:
                unique_ids.add(candidate)
                ordered_ids.append(candidate)

        for span in filtered_spans:
            if fact_name == "agent_sessions":
                _add(_attr(span, "scope.agentic.session"))
            elif fact_name == "agent_requests":
                if _span_type(span) == "agentic.turn":
                    _add(_attr(span, "scope.agentic.turn"))
            elif fact_name == "agent_operations":
                if _span_type(span) == "agentic.invocation":
                    _add(_attr(span, "scope.agentic.invocation"))
            elif fact_name == "tool_operations":
                if _span_type(span) == "agentic.tool.invocation":
                    _add(f"{_trace_id(span)}.{_span_id(span)}")
            elif fact_name == "inferences":
                if _span_type(span) == "inference":
                    _add(f"{_trace_id(span)}.{_span_id(span)}")
            elif fact_name == "traces":
                _add(_trace_id(span))
            elif fact_name == "conversations":
                _add(_attr(span, "scope.msteams.conversation.id"))
            elif fact_name == "git_commits":
                _add(_attr(span, "scope.git.commit.hash"))
            elif fact_name == "test_runs":
                _add(_attr(span, "scope.git.run.id"))
            elif fact_name == "tests":
                run_id = _attr(span, "scope.git.run.id")
                test_name = _attr(span, "scope.test_name")
                if run_id and test_name:
                    _add(f"{run_id}.{test_name}")

        return ordered_ids

    def _submit_eval_job(self, submit_url: str, headers: dict, payload: dict, params: dict) -> tuple[str, str, str, list[dict]]:
        """Submit one eval job and parse result payload."""
        try:
            response = requests.post(
                url=submit_url,
                headers=headers,
                json=payload,
                params=params
            )
        except requests.Timeout as exc:
            raise AssertionError(f"Evaluation service request timed out: {exc}") from exc
        except requests.RequestException as exc:
            raise AssertionError(f"Failed to reach evaluation service: {exc}") from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = response.status_code
            if status_code == 404:
                raise AssertionError(
                    "Trace not found in evaluation service. Confirm the span data was ingested before running check_eval."
                ) from exc
            response_body = response.text or "<empty body>"
            raise AssertionError(f"Evaluation service returned HTTP {status_code}: {response_body}") from exc

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" not in content_type:
            raise AssertionError(f"Evaluation service returned non-JSON response: {response.text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise AssertionError(f"Evaluation service returned invalid JSON: {response.text}") from exc

        try:
            job_id = data.get("job_id")
            eval_result = data.get("result")
            label = json.loads(eval_result[0].get('result')).get('label')
            explanation = json.loads(eval_result[0].get('result')).get('explanation')
        except Exception as exc:
            raise AssertionError(
                f"Unexpected response format from evaluation service. Expected 'result' key in response. Received: {data}"
            ) from exc

        return job_id, label, explanation, eval_result

    def evaluate(self, filtered_spans: Optional[list[Span]] = [], eval_name: Optional[str] = "", fact_name: Optional[str] = "traces", eval_args: dict = {}) -> Union[str, dict]:
        if not eval_name:
            raise ValueError("eval_name is required for evaluation.")
        
        if not filtered_spans:
            raise ValueError("No spans provided for evaluation.")
        
        # Validate and default fact_name if not provided
        if not fact_name:
            fact_name = "traces"
        
        # Map user-friendly fact name to internal name
        fact_name = self._map_fact_name(fact_name)
        
        logger.info(f"After mapping: fact_name={fact_name} (this is the internal DB name)")

        # Get API credentials
        api_key = (os.getenv("OKAHU_API_KEY") or "").strip()
        if not api_key:
            raise AssertionError("OKAHU_API_KEY is not configured.")

        # Export on first eval call only
        if not self._trace_exported:
            self.export_trace(filtered_spans)
        
        # Verify eval template exists before submitting job
        self.verify_eval_template_exists(eval_name=eval_name, fact_name=fact_name)

        # setting parameters, headers, payload for eval job submission
        trace_id = self._current_trace_id
        span = filtered_spans[0]
        workflow_name = span.attributes.get("workflow.name")
        base = os.getenv("OKAHU_EVALUATION_ENDPOINT", OKAHU_PROD_EVALUATION_ENDPOINT).rstrip("/")
        submit_url = f"{base}/v1/eval/jobs"
        start_span_ns = span.start_time - 24 * 60 * 60 * 1e9  # 24 hours before the first span's start time, in nanoseconds
        end_span_ns = span.end_time + 24 * 60 * 60 * 1e9  # 24 hours after the first span's end time, in nanoseconds
        start = datetime.fromtimestamp(start_span_ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        end = datetime.fromtimestamp(end_span_ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        fact_ids = self.enumerate_fact_ids(filtered_spans=filtered_spans, fact_name=fact_name)
        if not fact_ids:
            raise AssertionError(f"No fact IDs found in spans for fact_name='{fact_name}'.")

        headers = {"x-api-key": api_key}
        payload = {"template_name": eval_name}
        label = None
        explanation = ""

        for fact_id in fact_ids:
            params = {
                "workflow_name": workflow_name,
                "start_time": start,
                "end_time": end,
                "breakdown_filter": fact_name,
                "trace_id": fact_id,
                "fact_name": fact_name,
                "shadow_eval": True
            }
            
            logger.info(
                "Submitting evaluation job: eval_name=%s, fact_name=%s (internal), fact_id=%s",
                eval_name,
                fact_name,
                fact_id
            )
            
            logger.debug(f"Request params: {params}")
            
            # submit evaluation job to okahu and handle response/errors
            try:
                response = requests.post(
                    url=submit_url,
                    headers=headers,
                    json=payload,
                    params=params,
                    timeout=60
                )
            except requests.Timeout as exc:
                raise AssertionError(f"Evaluation service request timed out: {exc}") from exc
            except requests.RequestException as exc:
                raise AssertionError(f"Failed to reach evaluation service: {exc}") from exc
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                status_code = response.status_code
                if status_code == 404:
                    raise AssertionError(
                        "Trace not found in evaluation service. Confirm the span data was ingested before running check_eval."
                    ) from exc
                response_body = response.text or "<empty body>"
                raise AssertionError(f"Evaluation service returned HTTP {status_code}: {response_body}") from exc
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/json" not in content_type:
                raise AssertionError(f"Evaluation service returned non-JSON response: {response.text}")
            try:
                data = response.json()
            except ValueError as exc:
                raise AssertionError(f"Evaluation service returned invalid JSON: {response.text}") from exc
            try:
                job_id = data.get("job_id")
                eval_result = data.get("result") or []
                parsed = json.loads(eval_result[0].get("result"))
                label = parsed.get("label")
                explanation = parsed.get("explanation")
            except AssertionError:
                raise
            except Exception as exc:
                raise AssertionError(
                    f"Unexpected response format from evaluation service. Expected 'result' key in response. Received: {data}"
                ) from exc
            
            # Export eval results if okahu exporter is configured
            if "okahu" in (os.getenv("MONOCLE_EXPORTER", "")):
                with OkahuEvalResultExporter(api_key=api_key, endpoint=base) as result_exporter:
                    result_exporter.export_results(
                        job_id=job_id,
                        eval_result=eval_result,
                        template_name=eval_name,
                        fact_name=fact_name,
                        timeout=30
                    )

        return label, explanation
    
    def cleanup(self):
        """Clean up trace from Okahu evaluation service. Called once at test end."""
        if not self._trace_exported or not self._current_trace_id:
            return  # Nothing to clean up
        
        # Get API credentials
        api_key = (os.getenv("OKAHU_API_KEY") or "").strip()
        if not api_key:
            raise AssertionError("OKAHU_API_KEY is not configured.")
        
        trace_id = self._current_trace_id
        base = os.getenv("OKAHU_EVALUATION_ENDPOINT", OKAHU_PROD_EVALUATION_ENDPOINT).rstrip("/")

        try:
            with OkahuEvalResultExporter(api_key=api_key, endpoint=base) as result_exporter:
                result_exporter.delete_trace(trace_id=trace_id)
        except Exception:
            pass
        finally:
            # Reset state
            self._trace_exported = False
            self._current_trace_id = None
    