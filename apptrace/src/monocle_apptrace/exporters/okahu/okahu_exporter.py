import json
import logging
from typing import Callable, Optional, Sequence
import requests
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, ConsoleSpanExporter
from requests.exceptions import ReadTimeout
from monocle_apptrace.exporters.base_exporter import SpanExporterBase, serialize_span
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor
from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value

REQUESTS_SUCCESS_STATUS_CODES = (200, 202, 204)
OKAHU_PROD_INGEST_ENDPOINT = "https://ingest.okahu.co/api/v1/trace/ingest"

logger = logging.getLogger(__name__)


def _get_okahu_api_key() -> Optional[str]:
    return get_monocle_env_value("OKAHU_API_KEY")


def _get_monocle_exporter() -> Optional[str]:
    return get_monocle_env_value("MONOCLE_EXPORTER")


class OkahuSpanExporter(SpanExporterBase):
    def __init__(
            self,
            endpoint: Optional[str] = None,
            timeout: Optional[int] = None,
            session: Optional[requests.Session] = None,
            task_processor: ExportTaskProcessor = None,
            evaluate: Optional[bool] = False
    ):
        """Okahu exporter."""
        super().__init__()
        okahu_endpoint: str = get_monocle_env_value("OKAHU_INGESTION_ENDPOINT") or OKAHU_PROD_INGEST_ENDPOINT
        if evaluate:
            okahu_endpoint = okahu_endpoint.replace("/trace/ingest", "/eval/ingest")
        self.endpoint = endpoint or okahu_endpoint
        api_key: Optional[str] = _get_okahu_api_key()
        self._closed = False
        if not api_key:
            raise ValueError("OKAHU_API_KEY not set.")
        self.timeout = timeout or 15
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "x-api-key": api_key}
        )

        self.task_processor = task_processor or None
        if task_processor is not None:
            task_processor.start()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        # After the call to Shutdown subsequent calls to Export are
        # not allowed and should return a Failure result
        if not hasattr(self, 'session'):
            return self.exporter.export(spans)
        
        if self._closed:
            logger.warning("Exporter already shutdown, ignoring batch")
            return SpanExportResult.FAILURE
        
        span_list = {
            "batch": []
        }

        # append the batch object with all the spans object
        for span in spans:
            if self.skip_export(span):
                continue
            span_list["batch"].append(serialize_span(span))

        # if there are no spans to export after filtering, then return
        if len(span_list["batch"]) == 0:
            return
        
        # Calculate is_root_span by checking if any span has no parent
        is_root_span = any(not span.parent for span in spans)

        def send_spans_to_okahu(span_list_local=None, is_root=False):
            try:
                result = self.session.post(
                    url=self.endpoint,
                    data=json.dumps(span_list_local),
                    timeout=self.timeout,
                )
                if result.status_code not in REQUESTS_SUCCESS_STATUS_CODES:
                    logger.error(
                        "Traces cannot be uploaded; status code: %s, message %s",
                        result.status_code,
                        result.text,
                    )
                    return SpanExportResult.FAILURE
                logger.debug("spans successfully exported to okahu. Is root span: %s", is_root)
                return SpanExportResult.SUCCESS
            except ReadTimeout as e:
                logger.warning("Trace export timed out: %s", str(e))
                return SpanExportResult.FAILURE

        # if async task function is present, then push the request to asnc task
        if self.task_processor is not None and callable(self.task_processor.queue_task):
            self.task_processor.queue_task(
                send_spans_to_okahu,
                kwargs={'span_list_local': span_list, 'is_root': is_root_span},
                is_root_span=is_root_span
            )
            return SpanExportResult.SUCCESS
        return send_spans_to_okahu(span_list, is_root_span)

    def shutdown(self) -> None:
        if self._closed:
            logger.warning("Exporter already shutdown, ignoring call")
            return
        if hasattr(self, 'session'):
            self.session.close()
        self._closed = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
    