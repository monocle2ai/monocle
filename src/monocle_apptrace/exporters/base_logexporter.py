import json
import os
import logging
from typing import Sequence
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from monocle_apptrace.exporters.logging_config import logger

class BaseLogExporter(SpanExporter):
    """
    A base class for exporting Logs to remote storage (S3, Azure Blob, etc.)
    """
    def __init__(self, storage_client, storage_location: str, timeout: int = 30):
        self.storage_client = storage_client
        self.storage_location = storage_location
        self.timeout = timeout

        log_file_path = "../tests/logger.log"
        self.logger = logger

        if os.path.exists(log_file_path):
            with open(log_file_path, "w"):
                pass

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        span_list = {"batch": []}

        for span in spans:
            try:
                span_data = span.to_json()
                span_list["batch"].append(json.loads(span_data))
            except Exception as e:
                self.logger.warning("Failed to process span: %s", e)
                return SpanExportResult.FAILURE

        self.logger.info("Exporting span batch: %s", json.dumps(span_list, indent=2))
        # for exporting Logs
        return self.export_to_storage()

    def export_to_storage(self) -> SpanExportResult:
        """
        Implement this method in subclasses to handle the actual exporting to specific storage systems (S3, Azure).
        """
        raise NotImplementedError("export_to_storage method should be implemented in subclass")

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
