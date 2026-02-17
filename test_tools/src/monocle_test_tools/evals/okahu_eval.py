import logging
import os
from datetime import datetime, timezone
from opentelemetry.sdk.trace import Span
import requests
from monocle_apptrace.exporters.okahu import okahu_exporter
from monocle_test_tools.evals.base_eval import BaseEval
from typing import Optional, Union

logger = logging.getLogger(__name__)
OKAHU_PROD_INGEST_ENDPOINT = "https://ingest.okahu.co/api/v1/trace/ingest"
OKAHU_PROD_EVALUATION_ENDPOINT = "https://eval.okahu.co/api/v1/eval/jobs"

class OkahuEval(BaseEval):
    def __init__(self, **data):
        eval_options = data.get("eval_options")
        super().__init__(eval_options=eval_options)
    
    def evaluate(self, filtered_spans:Optional[list[Span]] = [],  eval_name:Optional[str] = "", fact_name: Optional[str] = "traces", eval_args: dict = {}) -> Union[str,dict]:
        if not eval_name:
            raise ValueError("eval_name is required for evaluation.")
        
        # fail if monocle exporter does not include okahu
        if "okahu" not in (os.getenv("MONOCLE_EXPORTER")):
            raise AssertionError("Current trace exporter is not configured to include Okahu. Please set MONOCLE_EXPORTER environment variable to include 'okahu' for evaluation features to work.")
        
        #find/set okahu api key, fail if not set
        api_key = (os.getenv("OKAHU_API_KEY")).strip()
        if not api_key:
            raise AssertionError("OKAHU_API_KEY is not configured.")

        # get traces, export them using okahu evaluator
        exporter = okahu_exporter.OkahuSpanExporter(evaluate=True)
        exporter.export(filtered_spans)
        # ensure all spans are flushed to Okahu before we proceed with the evaluation
        exporter.shutdown()

        #setting parameters, headers, payload for eval job submission
        base = os.getenv("OKAHU_EVALUATION_ENDPOINT", OKAHU_PROD_EVALUATION_ENDPOINT).rstrip("/")
        submit_url = f"{base}/v1/eval/jobs"

        span = filtered_spans[0]
        workflow_name = span.attributes.get("workflow.name")
        trace = format(span.get_span_context().trace_id, '032x')
        start_span_ns = span.start_time - 24 * 60 * 60 * 1e9  # 24 hours before the first span's start time, in nanoseconds
        end_span_ns = span.end_time + 24 * 60 * 60 * 1e9  # 24 hours after the first span's end time, in nanoseconds
        start = datetime.fromtimestamp(start_span_ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        end = datetime.fromtimestamp(end_span_ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        headers = {"x-api-key": api_key}
        payload = {"template_name": eval_name}
        params = {
            "workflow_name": workflow_name,
            "start_time": start,
            "end_time": end,
            "breakdown_filter": "traces",
            "trace_id": trace, 
            "fact_name": "traces",
            "shadow_eval": True
        }
        
        # submit evaluation job to okahu and handle response/errors
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
                raise AssertionError("Trace not found in evaluation service. Confirm the span data was ingested before running check_eval.") from exc
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
            eval_result = data.get("result")
        except Exception as exc:
            raise AssertionError(f"Unexpected response format from evaluation service. Expected 'result' key in response. Received: {data}") from exc
        
        # clear table after evaluation
        ingest = os.getenv("OKAHU_INGESTION_ENDPOINT", OKAHU_PROD_INGEST_ENDPOINT).rstrip("/")
        delete_url = ingest.replace("/trace/ingest", "/eval/delete")
        params = {"trace_id": trace}     
        try:
            response = requests.delete(delete_url, headers=headers, params=params)
            response.raise_for_status() 
            logging.info(f"Success: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"✗ Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"Status Code: {e.response.status_code}")
                logging.error(f"Response: {e.response.text}")

        return eval_result  