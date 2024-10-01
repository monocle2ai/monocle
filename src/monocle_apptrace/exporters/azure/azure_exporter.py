import os
import time
import random
import datetime
import logging
import asyncio
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient  # AIO version for async
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError, ServiceRequestError
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from typing import Sequence
from monocle_apptrace.exporters.span_exporter_base import SpanExporterBase

logger = logging.getLogger(__name__)

class AzureBlobSpanExporter(SpanExporterBase):
    def __init__(self, connection_string=None, container_name=None):
        super().__init__()
        DEFAULT_FILE_PREFIX = "monocle_trace_"
        DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
        self.max_batch_size = 500
        self.export_interval = 1
        # Use default values if none are provided
        if not connection_string:
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if not connection_string:
                raise ValueError("Azure Storage connection string is not provided or set in environment variables.")

        if not container_name:
            container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'default-container')

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

    def __container_exists(self, container_name):
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            container_client.get_container_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking if container {container_name} exists: {e}")
            return False

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
            span_data_list = [span.to_json() for span in spans]
            return "[" + ", ".join(span_data_list) + "]"
        except Exception as e:
            logger.error(f"Error serializing spans: {e}")
            raise

    async def __export_spans(self):
        if len(self.export_queue) == 0:
            return  # Nothing to export

        batch_to_export = self.export_queue[:self.max_batch_size]
        serialized_data = self.__serialize_spans(batch_to_export)
        self.export_queue = self.export_queue[self.max_batch_size:]
        try:
            if asyncio.get_event_loop().is_running():
                # Run asynchronously without closing the event loop
                task = asyncio.create_task(self._retry_with_backoff(self.__upload_to_blob, serialized_data))
                await task
            else:
                # If not in an event loop, use asyncio.run()
                await self._retry_with_backoff(self.__upload_to_blob, serialized_data)
        except Exception as e:
            logger.error(f"Failed to upload span batch: {e}")

    def __upload_to_blob(self, span_data_batch: str):
        current_time = datetime.datetime.now().strftime(self.time_format)
        file_name = f"{self.file_prefix}{current_time}.json"
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
        blob_client.upload_blob(span_data_batch, overwrite=True)  # Make sure this is an async call
        logger.info(f"Span batch uploaded to Azure Blob Storage as {file_name}.")

    async def force_flush(self, timeout_millis: int = 30000) -> bool:
        await self.__export_spans()
        return True

    def shutdown(self) -> None:
        logger.info("AzureBlobSpanExporter has been shut down.")
