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
        self._api_key = None
        self._base_url = None
    
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
        self._api_key = (os.getenv("OKAHU_API_KEY") or "").strip()
        if not self._api_key:
            raise AssertionError("OKAHU_API_KEY is not configured.")
        
        self._base_url = os.getenv("OKAHU_EVALUATION_ENDPOINT", OKAHU_PROD_EVALUATION_ENDPOINT).rstrip("/")
        
        # Export spans to Okahu
        exporter = okahu_exporter.OkahuSpanExporter(evaluate=True)
        exporter.export(filtered_spans)
        exporter.shutdown()
        
        self._trace_exported = True
        return trace_id
    
    def evaluate(self, filtered_spans:Optional[list[Span]] = [],  eval_name:Optional[str] = "", fact_name: Optional[str] = "traces", eval_args: dict = {}) -> Union[str,dict]:
        if not eval_name:
            raise ValueError("eval_name is required for evaluation.")
        
        if not filtered_spans:
            raise ValueError("No spans provided for evaluation.")
                
        # LAZY EXPORT: Export on first eval call only
        if not self._trace_exported:
            self.export_trace(filtered_spans)
        
        trace_id = self._current_trace_id
        span = filtered_spans[0]
        workflow_name = span.attributes.get("workflow.name")
        
        #setting parameters, headers, payload for eval job submission
        submit_url = f"{self._base_url}/v1/eval/jobs"
        start_span_ns = span.start_time - 24 * 60 * 60 * 1e9  # 24 hours before the first span's start time, in nanoseconds
        end_span_ns = span.end_time + 24 * 60 * 60 * 1e9  # 24 hours after the first span's end time, in nanoseconds
        start = datetime.fromtimestamp(start_span_ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        end = datetime.fromtimestamp(end_span_ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        headers = {"x-api-key": self._api_key}
        payload = {"template_name": eval_name}
        params = {
            "workflow_name": workflow_name,
            "start_time": start,
            "end_time": end,
            "breakdown_filter": "traces",
            "trace_id": trace_id, 
            "fact_name": "traces",
            "shadow_eval": True
        }
        
        # submit evaluation job to okahu and handle response/errors
        try:
            response = requests.post(
                url=submit_url,
                headers=headers,
                json=payload,
                params=params,
                timeout=30
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
        except Exception as exc:
            raise AssertionError(
                f"Unexpected response format from evaluation service. Expected 'result' key in response. Received: {data}"
            ) from exc
        
        # Export eval results if okahu exporter is configured
        if "okahu" in (os.getenv("MONOCLE_EXPORTER", "")):
            with OkahuEvalResultExporter(api_key=self._api_key, base_url=self._base_url) as result_exporter:
                result_exporter.export_results(
                    job_id=job_id,
                    eval_result=eval_result,
                    template_name=eval_name
                )
        
        return label
    
    def cleanup(self):
        """Clean up trace from Okahu evaluation service. Called once at test end."""
        if not self._trace_exported or not self._current_trace_id:
            return  # Nothing to clean up
        
        trace_id = self._current_trace_id
        
        try:
            with OkahuEvalResultExporter(api_key=self._api_key, base_url=self._base_url) as result_exporter:
                result_exporter.delete_trace(trace_id=trace_id)
        except Exception:
            pass
        finally:
            # Reset state
            self._trace_exported = False
            self._current_trace_id = None
    