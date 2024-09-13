import os
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError, ServiceRequestError
import datetime
from typing import Sequence
import logging
import time
import random

# Configuration
DEFAULT_FILE_PREFIX = "monocle_trace_"
DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
MAX_BATCH_SIZE = 500  # Max number of spans per batch
MAX_QUEUE_SIZE = 10000  # Max number of spans in the queue
EXPORT_INTERVAL = 1  # Maximum interval (in seconds) to wait before exporting spans
BACKOFF_FACTOR = 2
MAX_RETRIES = 10

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureBlobSpanExporter(SpanExporter):
    def __init__(self, connection_string, container_name):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
        self.file_prefix = DEFAULT_FILE_PREFIX
        self.time_format = DEFAULT_TIME_FORMAT
        self.export_queue = []
        self.last_export_time = time.time()

        # Check if container exists or create it
        if not self._container_exists(container_name):
            try:
                self.blob_service_client.create_container(container_name)
                logger.info(f"Container {container_name} created successfully.")
            except Exception as e:
                logger.error(f"Error creating container {container_name}: {e}")
                raise e

    def _container_exists(self, container_name):
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
        try:
            # Add spans to the export queue
            for span in spans:
                self.export_queue.append(span)
                # If the queue reaches MAX_BATCH_SIZE, export the spans
                if len(self.export_queue) >= MAX_BATCH_SIZE:
                    self._export_spans()

            # Check if it's time to force a flush
            current_time = time.time()
            if current_time - self.last_export_time >= EXPORT_INTERVAL:
                self._export_spans()  # Export spans if time interval has passed
                self.last_export_time = current_time  # Reset the last export time

            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Error exporting spans: {e}")
            return SpanExportResult.FAILURE

    def _serialize_spans(self, spans: Sequence[ReadableSpan]) -> str:
        try:
            # Serialize spans to JSON or any other format you prefer
            span_data_list = [span.to_json() for span in spans]
            return "[" + ", ".join(span_data_list) + "]"
        except Exception as e:
            logger.error(f"Error serializing spans: {e}")
            raise

    def _export_spans(self):
        if len(self.export_queue) == 0:
            return  # Nothing to export

        # Take a batch of spans from the queue
        batch_to_export = self.export_queue[:MAX_BATCH_SIZE]
        serialized_data = self._serialize_spans(batch_to_export)
        self.export_queue = self.export_queue[MAX_BATCH_SIZE:]  # Remove exported spans from the queue

        try:
            self._upload_to_blob_with_retry(serialized_data)
        except Exception as e:
            logger.error(f"Failed to upload span batch: {e}")

    def _upload_to_blob_with_retry(self, span_data_batch: str):
        current_time = datetime.datetime.now().strftime(self.time_format)
        file_name = f"{self.file_prefix}{current_time}.json"
        attempt = 0

        while attempt < MAX_RETRIES:
            try:
                blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
                blob_client.upload_blob(span_data_batch, overwrite=True)
                logger.info(f"Span batch uploaded to Azure Blob Storage as {file_name}.")
                return
            except ServiceRequestError as e:
                logger.warning(f"Network connectivity error: {e}. Retrying in {BACKOFF_FACTOR ** attempt} seconds...")
                sleep_time = BACKOFF_FACTOR * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
                attempt += 1
            except ClientAuthenticationError as e:
                logger.error(f"Failed to authenticate with Azure Blob Storage: {str(e)}")
                break  # Authentication errors should not be retried
            except Exception as e:
                logger.warning(f"Retry {attempt}/{MAX_RETRIES} failed due to: {str(e)}")
                sleep_time = BACKOFF_FACTOR * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Waiting for {sleep_time:.2f} seconds before retrying...")
                time.sleep(sleep_time)
                attempt += 1

        logger.error("Max retries exceeded. Failed to upload spans to Azure Blob Storage.")
        raise ServiceRequestError(message="Azure Blob Upload Endpoint")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._export_spans()  # Export any remaining spans in the queue
        return True

    def shutdown(self) -> None:
        logger.info("AzureBlobSpanExporter has been shut down.")


# import os
# from opentelemetry.sdk.trace import Span, ReadableSpan
# from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
# from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
# import grpc
# import datetime
# from concurrent import futures
# from typing import Optional, Callable, Sequence
# import threading
# import time
# import json
# from collections import deque
# import asyncio
#
# DEFAULT_FILE_PREFIX = "monocle_trace_"
# DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
# MAX_SPANS_PER_BATCH = 1000  # Maximum number of spans to batch
# MAX_BATCH_SIZE = 10000  # Maximum number of spans to queue for processing
# RETRY_COUNT = 3  # Number of times to retry on failure
# RETRY_DELAY = 2  # Initial delay for retries in seconds
# BATCH_TIMEOUT_SECONDS = 5  # Maximum time to wait before forcing a batch upload
#
#
# class AzureBlobSpanExporter(SpanExporter):
#     def __init__(self, connection_string, container_name):
#         self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#         self.container_name = container_name
#         self.file_prefix = DEFAULT_FILE_PREFIX
#         self.time_format = DEFAULT_TIME_FORMAT
#
#         # Create the container if it doesn't exist
#         container_client = self.blob_service_client.get_container_client(container_name)
#         if not container_client.exists():
#             container_client.create_container()
#
#         # Span queue using deque for efficient append/pop operations
#         self._span_queue = deque()
#         self._lock = threading.Lock()
#         self._stop_event = threading.Event()
#
#         # Event loop for asynchronous operations
#         self._event_loop = asyncio.new_event_loop()
#         self._export_task = self._event_loop.create_task(self._export_worker())
#
#         # Start the event loop in a separate thread
#         self._loop_thread = threading.Thread(target=self._run_event_loop)
#         self._loop_thread.daemon = True
#         self._loop_thread.start()
#
#     def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
#         with self._lock:
#             if len(self._span_queue) + len(spans) > MAX_BATCH_SIZE:
#                 print("Exporter queue is full; dropping spans")
#                 return SpanExportResult.FAILURE
#
#             self._span_queue.extend(spans)
#
#         return SpanExportResult.SUCCESS
#
#     def _run_event_loop(self):
#         asyncio.set_event_loop(self._event_loop)
#         self._event_loop.run_forever()
#
#     async def _export_worker(self):
#         while not self._stop_event.is_set():
#             await self._process_spans()
#             await asyncio.sleep(0.1)  # Sleep interval between processing batches
#
#     async def _process_spans(self):
#         spans_to_export = []
#         with self._lock:
#             if len(self._span_queue) == 0:
#                 return
#
#             # Take a batch of spans from the queue
#             spans_to_export = [self._span_queue.popleft() for _ in
#                                range(min(MAX_SPANS_PER_BATCH, len(self._span_queue)))]
#
#         # Serialize and upload spans asynchronously
#         if spans_to_export:
#             span_data_batch = await self._serialize_spans(spans_to_export)
#             await self._retry_upload(span_data_batch)
#
#     async def _serialize_spans(self, spans: list[ReadableSpan]) -> str:
#         # Serialize spans to JSON asynchronously
#         loop = asyncio.get_event_loop()
#         return await loop.run_in_executor(None, json.dumps, [span.to_json() for span in spans])
#
#     async def _retry_upload(self, span_data_batch: str):
#         retry_attempts = 0
#         while retry_attempts < RETRY_COUNT:
#             try:
#                 await self._upload_to_blob(span_data_batch)
#                 return
#             except Exception as e:
#                 print(f"Error uploading spans: {e}. Retrying in {RETRY_DELAY} seconds...")
#                 await asyncio.sleep(RETRY_DELAY * (2 ** retry_attempts))  # Exponential backoff
#                 retry_attempts += 1
#
#     async def _upload_to_blob(self, span_data_batch: str):
#         current_time = datetime.datetime.now().strftime(self.time_format)
#         blob_name = f"{self.file_prefix}{current_time}.json"
#         blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
#
#         # Asynchronous upload using run_in_executor
#         loop = asyncio.get_event_loop()
#         await loop.run_in_executor(None, blob_client.upload_blob, span_data_batch, True)
#         print(f"Span batch uploaded to Azure Blob Storage as {blob_name}.")
#
#     def force_flush(self, timeout_millis: int = 30000) -> bool:
#         self._stop_event.set()
#         self._event_loop.call_soon_threadsafe(self._event_loop.stop)
#         self._loop_thread.join(timeout_millis / 1000)
#         return True
#
#     def shutdown(self) -> None:
#         self.force_flush()

# import os
# from opentelemetry.sdk.trace import ReadableSpan
# from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
# from azure.storage.blob import BlobServiceClient, ContainerClient
# from azure.core.exceptions import ResourceExistsError, AzureError, ServiceRequestError, ClientAuthenticationError
# from requests.exceptions import RequestException
# import urllib3
# import datetime
# from typing import Sequence
# import logging
# import time
# import random
#
# # Configuration
# DEFAULT_FILE_PREFIX = "monocle_trace_"
# DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
# MAX_BATCH_SIZE = 500  # Max number of spans per batch
# BACKOFF_FACTOR = 2
# MAX_RETRIES = 10
#
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# class AzureBlobSpanExporter(SpanExporter):
#     def __init__(self, connection_string, container_name, max_retries=3, backoff_factor=2):
#         self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#         self.container_name = container_name
#         self.max_retries = max_retries
#         self.backoff_factor = backoff_factor
#         self.file_prefix = DEFAULT_FILE_PREFIX
#         self.time_format = DEFAULT_TIME_FORMAT
#
#         # Create the container if it doesn't exist
#         container_client = self.blob_service_client.get_container_client(container_name)
#         if not container_client.exists():
#             container_client.create_container()
#
#     def _upload_to_blob(self, span_data_batch: str):
#         current_time = datetime.datetime.now().strftime(self.time_format)
#         blob_name = f"{self.file_prefix}{current_time}.json"
#         blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
#
#         retries = 0
#         while retries < self.max_retries:
#             try:
#                 blob_client.upload_blob(span_data_batch, overwrite=True)
#                 print(f"Span batch uploaded to Azure Blob Storage as {blob_name}.")
#                 return
#             except Exception as e:
#                 logging.error(f"Attempt {retries + 1} failed with error: {e}")
#                 retries += 1
#                 time.sleep(self.backoff_factor ** retries)  # Exponential backoff
#         logging.error(f"Failed to upload span batch to Azure Blob Storage after {self.max_retries} attempts.")
#
#     def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
#         try:
#             span_data_batch = self._serialize_spans(spans)
#             self._upload_to_blob(span_data_batch)
#             return SpanExportResult.SUCCESS
#         except Exception as e:
#             print(f"Error exporting spans: {e}")
#             return SpanExportResult.FAILURE
#
#     def _serialize_spans(self, spans: list[ReadableSpan]) -> str:
#         # Serialize spans to JSON or any other format you prefer
#         # This example concatenates JSON representations of all spans into a single JSON array
#         span_data_list = [span.to_json() for span in spans]
#         return "[" + ", ".join(span_data_list) + "]"
#     def force_flush(self, timeout_millis: int = 30000) -> bool:
#         return True
#
#     def shutdown(self) -> None:
#         pass

