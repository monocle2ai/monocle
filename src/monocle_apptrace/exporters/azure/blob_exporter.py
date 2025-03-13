import os
import time
import random
import datetime
import logging
import asyncio
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError, ServiceRequestError
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from typing import Sequence, Optional
from monocle_apptrace.exporters.base_exporter import SpanExporterBase
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor
import json
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SDK_VERSION
logger = logging.getLogger(__name__)

class AzureBlobSpanExporter(SpanExporterBase):
    def __init__(self, connection_string=None, container_name=None, task_processor: Optional[ExportTaskProcessor] = None):
        super().__init__()
        DEFAULT_FILE_PREFIX = "monocle_trace_"
        DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
        self.max_batch_size = 500
        self.export_interval = 1
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
        # Add spans to the export queue
        for span in spans:
            # Azure blob library has a check to generate it's own span if OpenTelemetry is loaded and Azure trace package is installed (just pip install azure-trace-opentelemetry)
            # With Monocle,OpenTelemetry is always loaded. If the Azure trace package is installed, then it triggers the blob trace generation on every blob operation.
            # Thus, the Monocle span write ends up generating a blob span which again comes back to the exporter .. and would result in an infinite loop.
            # To avoid this, we check if the span has the Monocle SDK version attribute and skip it if it doesn't. That way the blob span genearted by Azure library are not exported.
            if not span.attributes.get(MONOCLE_SDK_VERSION):
                continue # TODO: All exporters to use same base class and check it there
            self.export_queue.append(span)
            if len(self.export_queue) >= self.max_batch_size:
                await self.__export_spans()

        # Force a flush if the interval has passed
        current_time = time.time()
        if current_time - self.last_export_time >= self.export_interval:
            await self.__export_spans()
            self.last_export_time = current_time

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

    async def __export_spans(self):
        if len(self.export_queue) == 0:
            return

        batch_to_export = self.export_queue[:self.max_batch_size]
        serialized_data = self.__serialize_spans(batch_to_export)
        self.export_queue = self.export_queue[self.max_batch_size:]
        
        # Calculate is_root_span by checking if any span has no parent
        is_root_span = any(not span.parent for span in batch_to_export)
        
        if self.task_processor is not None and callable(getattr(self.task_processor, 'queue_task', None)):
            self.task_processor.queue_task(self.__upload_to_blob, serialized_data, is_root_span)
        else:
            try:
                self.__upload_to_blob(serialized_data, is_root_span)
            except Exception as e:
                logger.error(f"Failed to upload span batch: {e}")

    @SpanExporterBase.retry_with_backoff(exceptions=(ResourceNotFoundError, ClientAuthenticationError, ServiceRequestError))
    def __upload_to_blob(self, span_data_batch: str, is_root_span: bool = False):
        current_time = datetime.datetime.now().strftime(self.time_format)
        file_name = f"{self.file_prefix}{current_time}.ndjson"
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
        blob_client.upload_blob(span_data_batch, overwrite=True)
        logger.info(f"Span batch uploaded to Azure Blob Storage as {file_name}. Is root span: {is_root_span}")

    async def force_flush(self, timeout_millis: int = 30000) -> bool:
        await self.__export_spans()
        return True

    def shutdown(self) -> None:
        if hasattr(self, 'task_processor') and self.task_processor is not None:
            self.task_processor.stop()
        logger.info("AzureBlobSpanExporter has been shut down.")
