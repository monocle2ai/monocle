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
    def __init__(self, connection_string=None, container_name=None):
        # Use default values if none are provided
        if connection_string is None:
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if connection_string is None:
                raise ValueError("Azure Storage connection string is not provided or set in environment variables.")

        if container_name is None:
            container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME','default-container')  # Use default container name if not provided

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


