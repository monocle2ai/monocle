import os
import datetime
import logging
import asyncio
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError, ServiceRequestError
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from typing import Sequence, Optional, Dict, List, Tuple
from monocle_apptrace.exporters.base_exporter import SpanExporterBase, format_trace_id_without_0x
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor
import json
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SDK_VERSION
logger = logging.getLogger(__name__)

HANDLE_TIMEOUT_SECONDS = 60  # 1 minute timeout for orphaned traces

class AzureBlobSpanExporter(SpanExporterBase):
    def __init__(self, connection_string=None, container_name=None, task_processor: Optional[ExportTaskProcessor] = None):
        super().__init__()
        DEFAULT_FILE_PREFIX = "monocle_trace_"
        DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
        self.max_batch_size = 500
        self.export_interval = 1
        # Dictionary to store spans by trace_id: {trace_id: (spans_list, creation_time, has_root_span)}
        self.trace_spans: Dict[int, Tuple[List[ReadableSpan], datetime.datetime, bool]] = {}
        # Use default values if none are provided
        if not connection_string:
            connection_string = os.getenv('MONOCLE_BLOB_CONNECTION_STRING')
            if not connection_string:
                raise ValueError("Azure Storage connection string is not provided or set in environment variables.")

        if not container_name:
            container_name = os.getenv('MONOCLE_BLOB_CONTAINER_NAME', 'default-container')

        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
        self.file_prefix = DEFAULT_FILE_PREFIX
        self.time_format = DEFAULT_TIME_FORMAT

        # Check if container exists or create it
        if not self.__container_exists(container_name):
            try:
                self.blob_service_client.create_container(container_name)
                logger.info(f"Container {container_name} created successfully.")
            except Exception as e:
                logger.error(f"Error creating container {container_name}: {e}")
                raise e

        self.task_processor = task_processor
        if self.task_processor is not None:
            self.task_processor.start()

    def __container_exists(self, container_name):
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            container_client.get_container_properties()
            return True
        except ResourceNotFoundError:
            logger.error(f"Container {container_name} not found (404).")
            return False
        except ClientAuthenticationError:
            logger.error(f"Access to container {container_name} is forbidden (403).")
            raise PermissionError(f"Access to container {container_name} is forbidden.")
        except Exception as e:
            logger.error(f"Unexpected error when checking if container {container_name} exists: {e}")
            raise e

    def _cleanup_expired_traces(self) -> None:
        """Upload and remove traces that have exceeded the timeout."""
        current_time = datetime.datetime.now()
        expired_trace_ids = []
        
        for trace_id, (spans, creation_time, _) in self.trace_spans.items():
            if (current_time - creation_time).total_seconds() > HANDLE_TIMEOUT_SECONDS:
                expired_trace_ids.append(trace_id)
        
        for trace_id in expired_trace_ids:
            self._upload_trace(trace_id)

    def _add_spans_to_trace(self, trace_id: int, spans: List[ReadableSpan], has_root: bool = False) -> None:
        """Add spans to a trace buffer, creating it if needed."""
        if trace_id in self.trace_spans:
            existing_spans, creation_time, existing_root = self.trace_spans[trace_id]
            existing_spans.extend(spans)
            has_root = has_root or existing_root
            self.trace_spans[trace_id] = (existing_spans, creation_time, has_root)
        else:
            self.trace_spans[trace_id] = (spans.copy(), datetime.datetime.now(), has_root)

    def _upload_trace(self, trace_id: int) -> None:
        """Upload a specific trace to Azure Blob and remove it from the buffer."""
        if trace_id not in self.trace_spans:
            return
        
        spans, _, _ = self.trace_spans[trace_id]
        if len(spans) == 0:
            del self.trace_spans[trace_id]
            return
        
        serialized_data = self.__serialize_spans(spans)
        if serialized_data:
            try:
                self.__upload_to_blob_with_trace_id(serialized_data, trace_id)
            except Exception as e:
                logger.error(f"Failed to upload trace {format_trace_id_without_0x(trace_id)}: {e}")
        
        del self.trace_spans[trace_id]

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Synchronous export method that internally handles async logic."""
        try:
            # Run the asynchronous export logic in an event loop
            asyncio.run(self._export_async(spans))
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Error exporting spans: {e}")
            return SpanExportResult.FAILURE

    async def _export_async(self, spans: Sequence[ReadableSpan]):
        """The actual async export logic is run here."""
        try:
            # Cleanup expired traces first
            self._cleanup_expired_traces()
            
            # Group spans by trace_id
            spans_by_trace = {}
            root_span_traces = set()
            
            for span in spans:
                # Azure blob library has a check to generate it's own span if OpenTelemetry is loaded and Azure trace package is installed (just pip install azure-trace-opentelemetry)
                # With Monocle,OpenTelemetry is always loaded. If the Azure trace package is installed, then it triggers the blob trace generation on every blob operation.
                # Thus, the Monocle span write ends up generating a blob span which again comes back to the exporter .. and would result in an infinite loop.
                # To avoid this, we check if the span has the Monocle SDK version attribute and skip it if it doesn't. That way the blob span genearted by Azure library are not exported.
                if self.skip_export(span):
                    continue
                
                trace_id = span.context.trace_id
                if trace_id not in spans_by_trace:
                    spans_by_trace[trace_id] = []
                spans_by_trace[trace_id].append(span)
                
                # Check if this span is a root span (no parent)
                if not span.parent:
                    root_span_traces.add(trace_id)
            
            # Add spans to their respective trace buffers
            for trace_id, trace_spans in spans_by_trace.items():
                has_root = trace_id in root_span_traces
                self._add_spans_to_trace(trace_id, trace_spans, has_root)
            
            # Upload complete traces (those with root spans)
            for trace_id in root_span_traces:
                if self.task_processor is not None and callable(getattr(self.task_processor, 'queue_task', None)):
                    # Queue the upload task
                    if trace_id in self.trace_spans:
                        spans_to_upload, _, _ = self.trace_spans[trace_id]
                        serialized_data = self.__serialize_spans(spans_to_upload)
                        if serialized_data:
                            self.task_processor.queue_task(
                                self.__upload_to_blob_with_trace_id,
                                kwargs={'span_data_batch': serialized_data, 'trace_id': trace_id},
                                is_root_span=True
                            )
                        del self.trace_spans[trace_id]
                else:
                    self._upload_trace(trace_id)
        except Exception as e:
            logger.error(f"Error in _export_async: {e}")

    def __serialize_spans(self, spans: Sequence[ReadableSpan]) -> str:
        try:
            valid_json_list = []
            for span in spans:
                try:
                    valid_json_list.append(span.to_json(indent=0).replace("\n", ""))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON format in span data: {span.context.span_id}. Error: {e}")
                    continue

            ndjson_data = "\n".join(valid_json_list) + "\n"
            return ndjson_data
        except Exception as e:
            logger.warning(f"Error serializing spans: {e}")
            return ""

    @SpanExporterBase.retry_with_backoff(exceptions=(ResourceNotFoundError, ClientAuthenticationError, ServiceRequestError))
    def __upload_to_blob_with_trace_id(self, span_data_batch: str, trace_id: int):
        """Upload spans for a specific trace to Azure Blob with trace ID in filename."""
        current_time = datetime.datetime.now().strftime(self.time_format)
        file_name = f"{self.file_prefix}{current_time}_{format_trace_id_without_0x(trace_id)}.ndjson"
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
        blob_client.upload_blob(span_data_batch, overwrite=True)
        logger.debug(f"Trace {format_trace_id_without_0x(trace_id)} uploaded to Azure Blob Storage as {file_name}.")

    async def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush all pending traces to Azure Blob."""
        trace_ids_to_upload = list(self.trace_spans.keys())
        for trace_id in trace_ids_to_upload:
            self._upload_trace(trace_id)
        return True

    def shutdown(self) -> None:
        """Upload all pending traces and shutdown."""
        # Upload all remaining traces
        trace_ids_to_upload = list(self.trace_spans.keys())
        for trace_id in trace_ids_to_upload:
            self._upload_trace(trace_id)
        
        if hasattr(self, 'task_processor') and self.task_processor is not None:
            self.task_processor.stop()
        logger.info("AzureBlobSpanExporter has been shut down.")
