import os
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
import boto3
from botocore.exceptions import EndpointConnectionError, NoCredentialsError, ClientError
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


class S3SpanExporter(SpanExporter):
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, region_name="us-east-1"):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.bucket_name = bucket_name
        self.file_prefix = DEFAULT_FILE_PREFIX
        self.time_format = DEFAULT_TIME_FORMAT
        self.export_queue = []
        self.last_export_time = time.time()

        # Check if bucket exists or create it
        if not self._bucket_exists(bucket_name):
            try:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region_name}
                )
                logger.info(f"Bucket {bucket_name} created successfully.")
            except ClientError as e:
                logger.error(f"Error creating bucket {bucket_name}: {e}")
                raise e

    def _bucket_exists(self, bucket_name):
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except self.s3_client.exceptions.NoSuchBucket:
            return False
        except Exception as e:
            logger.error(f"Error checking if bucket {bucket_name} exists: {e}")
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
            self._upload_to_s3_with_retry(serialized_data)
        except Exception as e:
            logger.error(f"Failed to upload span batch: {e}")

    def _upload_to_s3_with_retry(self, span_data_batch: str):
        current_time = datetime.datetime.now().strftime(self.time_format)
        file_name = f"{self.file_prefix}{current_time}.json"
        attempt = 0

        while attempt < MAX_RETRIES:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=file_name,
                    Body=span_data_batch
                )
                logger.info(f"Span batch uploaded to AWS S3 as {file_name}.")
                return
            except EndpointConnectionError as e:
                logger.warning(f"Network connectivity error: {e}. Retrying in {BACKOFF_FACTOR ** attempt} seconds...")
                sleep_time = BACKOFF_FACTOR * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
                attempt += 1
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("RequestTimeout", "ThrottlingException", "InternalError", "ServiceUnavailable"):
                    logger.warning(f"Retry {attempt}/{MAX_RETRIES} failed due to network issue: {str(e)}")
                else:
                    logger.error(f"Failed to upload trace data: {str(e)}")
                    break  # For other types of errors, do not retry

                sleep_time = BACKOFF_FACTOR * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Waiting for {sleep_time:.2f} seconds before retrying...")
                time.sleep(sleep_time)
                attempt += 1
            except (NoCredentialsError, ClientError) as e:
                logger.error(f"Error uploading spans to S3: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error uploading spans to S3: {e}")
                raise

        logger.error("Max retries exceeded. Failed to upload spans to S3.")
        raise EndpointConnectionError(endpoint_url="S3 Upload Endpoint")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._export_spans()  # Export any remaining spans in the queue
        return True

    def shutdown(self) -> None:
        logger.info("S3SpanExporter has been shut down.")


# import os
# from opentelemetry.sdk.trace import ReadableSpan
# from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
# import boto3
# from botocore.exceptions import EndpointConnectionError, NoCredentialsError, ClientError
# import datetime
# from concurrent import futures
# from typing import Sequence
# import logging
# import time
# import random
# import threading
# from queue import Queue
#
# # Configuration
# DEFAULT_FILE_PREFIX = "monocle_trace_"
# DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
# MAX_BATCH_SIZE = 500  # Max number of spans per batch
# MAX_QUEUE_SIZE = 10000  # Max number of batches in the queue
# NUM_WORKER_THREADS = 4  # Number of worker threads for concurrent uploads
#
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# class S3SpanExporter(SpanExporter):
#     def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, region_name="us-east-1", max_retries=10,
#                  backoff_factor=2):
#         self.s3_client = boto3.client(
#             's3',
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             region_name=region_name
#         )
#         self.bucket_name = bucket_name
#         self.file_prefix = DEFAULT_FILE_PREFIX
#         self.time_format = DEFAULT_TIME_FORMAT
#         self.max_retries = max_retries
#         self.backoff_factor = backoff_factor
#         self.export_queue = Queue(maxsize=MAX_QUEUE_SIZE)
#
#         # Check if bucket exists or create it
#         if not self._bucket_exists(bucket_name):
#             try:
#                 self.s3_client.create_bucket(
#                     Bucket=bucket_name,
#                     CreateBucketConfiguration={'LocationConstraint': region_name}
#                 )
#                 logger.info(f"Bucket {bucket_name} created successfully.")
#             except ClientError as e:
#                 logger.error(f"Error creating bucket {bucket_name}: {e}")
#                 raise e
#
#         # Start worker threads for concurrent uploads
#         self.worker_threads = []
#         for _ in range(NUM_WORKER_THREADS):
#             worker = threading.Thread(target=self._upload_worker)
#             worker.daemon = True
#             worker.start()
#             self.worker_threads.append(worker)
#
#     def _bucket_exists(self, bucket_name):
#         try:
#             self.s3_client.head_bucket(Bucket=bucket_name)
#             return True
#         except self.s3_client.exceptions.NoSuchBucket:
#             return False
#         except Exception as e:
#             logger.error(f"Error checking if bucket {bucket_name} exists: {e}")
#             return False
#
#     def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
#         try:
#             # Split spans into batches
#             for i in range(0, len(spans), MAX_BATCH_SIZE):
#                 span_batch = spans[i:i + MAX_BATCH_SIZE]
#                 if len(span_batch) > 0:
#                     serialized_data = self._serialize_spans(span_batch)
#                     self.export_queue.put(serialized_data)
#                     logger.info(f"Batch of {len(span_batch)} spans added to the queue for export.")
#
#             return SpanExportResult.SUCCESS
#         except Exception as e:
#             logger.error(f"Error exporting spans: {e}")
#             return SpanExportResult.FAILURE
#
#     def _serialize_spans(self, spans: Sequence[ReadableSpan]) -> str:
#         try:
#             # Serialize spans to JSON or any other format you prefer
#             span_data_list = [span.to_json() for span in spans]
#             return "[" + ", ".join(span_data_list) + "]"
#         except Exception as e:
#             logger.error(f"Error serializing spans: {e}")
#             raise
#
#     def _upload_worker(self):
#         while True:
#             span_data_batch = self.export_queue.get()
#             if span_data_batch is None:
#                 break
#             try:
#                 self._upload_to_s3_with_retry(span_data_batch)
#             except Exception as e:
#                 logger.error(f"Failed to upload span batch: {e}")
#             finally:
#                 self.export_queue.task_done()
#
#     def _upload_to_s3_with_retry(self, span_data_batch: str):
#         current_time = datetime.datetime.now().strftime(self.time_format)
#         file_name = f"{self.file_prefix}{current_time}.json"
#         attempt = 0
#
#         while attempt < self.max_retries:
#             try:
#                 self.s3_client.put_object(
#                     Bucket=self.bucket_name,
#                     Key=file_name,
#                     Body=span_data_batch
#                 )
#                 logger.info(f"Span batch uploaded to AWS S3 as {file_name}.")
#                 return
#             except EndpointConnectionError as e:
#                 logger.warning(
#                     f"Network connectivity error: {e}. Retrying in {self.backoff_factor ** attempt} seconds...")
#                 sleep_time = self.backoff_factor * (2 ** attempt) + random.uniform(0, 1)
#                 time.sleep(sleep_time)
#                 attempt += 1
#             except ClientError as e:
#                 error_code = e.response.get("Error", {}).get("Code", "")
#                 if error_code in ("RequestTimeout", "ThrottlingException", "InternalError", "ServiceUnavailable"):
#                     logger.warning(f"Retry {attempt}/{self.max_retries} failed due to network issue: {str(e)}")
#                 else:
#                     logger.error(f"Failed to upload trace data: {str(e)}")
#                     break  # For other types of errors, do not retry
#
#                 sleep_time = self.backoff_factor * (2 ** attempt) + random.uniform(0, 1)
#                 logger.info(f"Waiting for {sleep_time:.2f} seconds before retrying...")
#                 time.sleep(sleep_time)
#                 attempt += 1
#             except (NoCredentialsError, ClientError) as e:
#                 logger.error(f"Error uploading spans to S3: {e}")
#                 raise
#             except Exception as e:
#                 logger.error(f"Unexpected error uploading spans to S3: {e}")
#                 raise
#
#         logger.error("Max retries exceeded. Failed to upload spans to S3.")
#         raise EndpointConnectionError(endpoint_url="S3 Upload Endpoint")
#
#     def force_flush(self, timeout_millis: int = 30000) -> bool:
#         self.export_queue.join()  # Wait for all queued tasks to be processed
#         return True
#
#     def shutdown(self) -> None:
#         # Stop worker threads
#         for _ in self.worker_threads:
#             self.export_queue.put(None)
#         for worker in self.worker_threads:
#             worker.join()
#         logger.info("S3SpanExporter has been shut down.")

# import os
# from opentelemetry.sdk.trace import Span, ReadableSpan
# from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
# import boto3
# from botocore.exceptions import EndpointConnectionError, NoCredentialsError, ClientError
# import grpc
# import datetime
# from concurrent import futures
# from typing import Optional, Callable, Sequence
# import logging
# import time
# import random
# import threading
# from queue import Queue
#
# # Configuration
# DEFAULT_FILE_PREFIX = "monocle_trace_"
# DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
# MAX_BATCH_SIZE = 500  # Max number of spans per batch
# MAX_QUEUE_SIZE = 10000  # Max number of batches in the queue
# NUM_WORKER_THREADS = 4  # Number of worker threads for concurrent uploads
#
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# class S3SpanExporter(SpanExporter):
#     def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, region_name="us-east-1", max_retries=10,
#                  backoff_factor=2):
#         self.s3_client = boto3.client(
#             's3',
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             region_name=region_name
#         )
#         self.bucket_name = bucket_name
#         self.file_prefix = DEFAULT_FILE_PREFIX
#         self.time_format = DEFAULT_TIME_FORMAT
#         self.max_retries = max_retries
#         self.backoff_factor = backoff_factor
#         self.export_queue = Queue(maxsize=MAX_QUEUE_SIZE)
#
#         # Check if bucket exists or create it
#         if not self._bucket_exists(bucket_name):
#             try:
#                 self.s3_client.create_bucket(
#                     Bucket=bucket_name,
#                     CreateBucketConfiguration={'LocationConstraint': region_name}
#                 )
#                 logger.info(f"Bucket {bucket_name} created successfully.")
#             except ClientError as e:
#                 logger.error(f"Error creating bucket {bucket_name}: {e}")
#                 raise e
#
#         # Start worker threads for concurrent uploads
#         self.worker_threads = []
#         for _ in range(NUM_WORKER_THREADS):
#             worker = threading.Thread(target=self._upload_worker)
#             worker.daemon = True
#             worker.start()
#             self.worker_threads.append(worker)
#
#     def _bucket_exists(self, bucket_name):
#         try:
#             self.s3_client.head_bucket(Bucket=bucket_name)
#             return True
#         except self.s3_client.exceptions.NoSuchBucket:
#             return False
#         except Exception as e:
#             logger.error(f"Error checking if bucket {bucket_name} exists: {e}")
#             return False
#
#     def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
#         try:
#             # Split spans into batches
#             for i in range(0, len(spans), MAX_BATCH_SIZE):
#                 span_batch = spans[i:i + MAX_BATCH_SIZE]
#                 serialized_data = self._serialize_spans(span_batch)
#                 self.export_queue.put(serialized_data)
#
#             return SpanExportResult.SUCCESS
#         except Exception as e:
#             logger.error(f"Error exporting spans: {e}")
#             return SpanExportResult.FAILURE
#
#     def _serialize_spans(self, spans: Sequence[ReadableSpan]) -> str:
#         try:
#             # Serialize spans to JSON or any other format you prefer
#             span_data_list = [span.to_json() for span in spans]
#             return "[" + ", ".join(span_data_list) + "]"
#         except Exception as e:
#             logger.error(f"Error serializing spans: {e}")
#             raise
#
#     def _upload_worker(self):
#         while True:
#             span_data_batch = self.export_queue.get()
#             if span_data_batch is None:
#                 break
#             try:
#                 self._upload_to_s3_with_retry(span_data_batch)
#             except Exception as e:
#                 logger.error(f"Failed to upload span batch: {e}")
#             finally:
#                 self.export_queue.task_done()
#
#     def _upload_to_s3_with_retry(self, span_data_batch: str):
#         current_time = datetime.datetime.now().strftime(self.time_format)
#         file_name = f"{self.file_prefix}{current_time}.json"
#         attempt = 0
#
#         while attempt < self.max_retries:
#             try:
#                 self.s3_client.put_object(
#                     Bucket=self.bucket_name,
#                     Key=file_name,
#                     Body=span_data_batch
#                 )
#                 logger.info(f"Span batch uploaded to AWS S3 as {file_name}.")
#                 return
#             except EndpointConnectionError as e:
#                 logger.warning(
#                     f"Network connectivity error: {e}. Retrying in {self.backoff_factor ** attempt} seconds...")
#                 sleep_time = self.backoff_factor * (2 ** attempt) + random.uniform(0, 1)
#                 time.sleep(sleep_time)
#                 attempt += 1
#             except ClientError as e:
#                 error_code = e.response.get("Error", {}).get("Code", "")
#                 if error_code in ("RequestTimeout", "ThrottlingException", "InternalError", "ServiceUnavailable"):
#                     logger.warning(f"Retry {attempt}/{self.max_retries} failed due to network issue: {str(e)}")
#                 else:
#                     logger.error(f"Failed to upload trace data: {str(e)}")
#                     break  # For other types of errors, do not retry
#
#                 sleep_time = self.backoff_factor * (2 ** attempt) + random.uniform(0, 1)
#                 logger.info(f"Waiting for {sleep_time:.2f} seconds before retrying...")
#                 time.sleep(sleep_time)
#                 attempt += 1
#             except (NoCredentialsError, ClientError) as e:
#                 logger.error(f"Error uploading spans to S3: {e}")
#                 raise
#             except Exception as e:
#                 logger.error(f"Unexpected error uploading spans to S3: {e}")
#                 raise
#
#         logger.error("Max retries exceeded. Failed to upload spans to S3.")
#         raise EndpointConnectionError(endpoint_url="S3 Upload Endpoint")
#
#     def force_flush(self, timeout_millis: int = 30000) -> bool:
#         self.export_queue.join()  # Wait for all queued tasks to be processed
#         return True
#
#     def shutdown(self) -> None:
#         # Stop worker threads
#         for _ in self.worker_threads:
#             self.export_queue.put(None)
#         for worker in self.worker_threads:
#             worker.join()
#         logger.info("S3SpanExporter has been shut down.")

# import os
# import json
# import threading
# import queue
# from opentelemetry.sdk.trace import Span, ReadableSpan
# from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
# import boto3
# from botocore.exceptions import EndpointConnectionError, NoCredentialsError, ClientError
# import grpc
# import datetime
# from concurrent import futures
# from typing import Optional, Callable, Sequence
# import logging
# import time
# import random
#
# # Configuration
# DEFAULT_FILE_PREFIX = "monocle_trace_"
# DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
# BATCH_SIZE = 1000  # Adjust batch size as needed
# MAX_WORKERS = 10  # Adjust the number of workers for parallel export
#
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# class S3SpanExporter(SpanExporter):
#     def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, region_name="us-east-1", max_retries=10, backoff_factor=2):
#         self.s3_client = boto3.client(
#             's3',
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             region_name=region_name
#         )
#         self.bucket_name = bucket_name
#         self.file_prefix = DEFAULT_FILE_PREFIX
#         self.time_format = DEFAULT_TIME_FORMAT
#         self.max_retries = max_retries
#         self.backoff_factor = backoff_factor
#
#         # Queue for managing span batches
#         self.span_queue = queue.Queue()
#         self._export_threads = []
#         self._stop_event = threading.Event()
#
#         # Start background threads for exporting spans
#         self._start_export_threads()
#
#         # Check if bucket exists or create it
#         if not self._bucket_exists(bucket_name):
#             try:
#                 self.s3_client.create_bucket(
#                     Bucket=bucket_name,
#                     CreateBucketConfiguration={'LocationConstraint': region_name}
#                 )
#                 logger.info(f"Bucket {bucket_name} created successfully.")
#             except ClientError as e:
#                 logger.error(f"Error creating bucket {bucket_name}: {e}")
#                 raise e
#
#     def _start_export_threads(self):
#         for _ in range(MAX_WORKERS):
#             thread = threading.Thread(target=self._export_worker)
#             thread.daemon = True
#             thread.start()
#             self._export_threads.append(thread)
#
#     def _export_worker(self):
#         while not self._stop_event.is_set():
#             try:
#                 span_batch = self.span_queue.get(timeout=1)
#                 if span_batch:
#                     self._upload_to_s3_with_retry(span_batch)
#                     self.span_queue.task_done()
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 logger.error(f"Error in export worker: {e}")
#
#     def _bucket_exists(self, bucket_name):
#         try:
#             self.s3_client.head_bucket(Bucket=bucket_name)
#             return True
#         except self.s3_client.exceptions.NoSuchBucket:
#             return False
#         except Exception as e:
#             logger.error(f"Error checking if bucket {bucket_name} exists: {e}")
#             return False
#
#     def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
#         try:
#             # Group spans into batches
#             for i in range(0, len(spans), BATCH_SIZE):
#                 span_batch = spans[i:i + BATCH_SIZE]
#                 serialized_batch = self._serialize_spans(span_batch)
#                 self.span_queue.put(serialized_batch)
#             return SpanExportResult.SUCCESS
#         except Exception as e:
#             logger.error(f"Error exporting spans: {e}")
#             return SpanExportResult.FAILURE
#
#     def _serialize_spans(self, spans: Sequence[ReadableSpan]) -> str:
#         try:
#             # Serialize spans to JSON or any other format you prefer
#             span_data_list = [span.to_json() for span in spans]
#             return "[" + ", ".join(span_data_list) + "]"
#         except Exception as e:
#             logger.error(f"Error serializing spans: {e}")
#             raise
#
#     def _upload_to_s3_with_retry(self, span_data_batch: str):
#         current_time = datetime.datetime.now().strftime(self.time_format)
#         file_name = f"{self.file_prefix}{current_time}.json"
#         attempt = 0
#
#         while attempt < self.max_retries:
#             try:
#                 self.s3_client.put_object(
#                     Bucket=self.bucket_name,
#                     Key=file_name,
#                     Body=span_data_batch
#                 )
#                 logger.info(f"Span batch uploaded to AWS S3 as {file_name}.")
#                 return
#             except EndpointConnectionError as e:
#                 # Handle network connectivity issues
#                 logger.warning(f"Network connectivity error: {e}. Retrying in {self.backoff_factor ** attempt} seconds...")
#                 sleep_time = self.backoff_factor * (2 ** attempt) + random.uniform(0, 1)
#                 time.sleep(sleep_time)
#                 attempt += 1
#             except ClientError as e:
#                 # Check if the error is due to connectivity issues
#                 error_code = e.response.get("Error", {}).get("Code", "")
#                 if error_code in ("RequestTimeout", "ThrottlingException", "InternalError", "ServiceUnavailable"):
#                     logger.warning(f"Retry {attempt}/{self.max_retries} failed due to network issue: {str(e)}")
#                 else:
#                     logger.error(f"Failed to upload trace data: {str(e)}")
#                     break  # For other types of errors, do not retry
#
#                 sleep_time = self.backoff_factor * (2 ** attempt) + random.uniform(0, 1)
#                 logger.warning(f"Waiting for {sleep_time:.2f} seconds before retrying...")
#                 time.sleep(sleep_time)
#                 attempt += 1
#             except (NoCredentialsError, ClientError) as e:
#                 # Handle AWS-related errors that are not recoverable
#                 logger.error(f"Error uploading spans to S3: {e}")
#                 raise
#             except Exception as e:
#                 # Catch-all for any other exceptions
#                 logger.error(f"Unexpected error uploading spans to S3: {e}")
#                 raise
#
#         # If all retries fail, log failure and raise an exception
#         logger.error("Max retries exceeded. Failed to upload spans to S3.")
#         raise EndpointConnectionError(endpoint_url="S3 Upload Endpoint")
#
#     def force_flush(self, timeout_millis: int = 30000) -> bool:
#         # Wait for all queued batches to be processed
#         self.span_queue.join()
#         return True
#
#     def shutdown(self) -> None:
#         # Signal all threads to stop
#         self._stop_event.set()
#         # Wait for all threads to finish
#         for thread in self._export_threads:
#             thread.join()


# import os
# from opentelemetry.sdk.trace import Span, ReadableSpan
# from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
# import boto3
# from botocore.exceptions import EndpointConnectionError, NoCredentialsError, ClientError
# import asyncio
# import datetime
# import random
# from concurrent.futures import ThreadPoolExecutor
# from queue import Queue
# from typing import Optional, Callable, Sequence
# import logging
# import time
#
# # Configuration
# DEFAULT_FILE_PREFIX = "monocle_trace_"
# DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
# MAX_BATCH_SIZE = 1000  # Max number of spans per batch
# UPLOAD_WORKER_COUNT = 5  # Number of concurrent upload workers
#
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# class S3SpanExporter(SpanExporter):
#     def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, region_name="us-east-1", max_retries=10,
#                  backoff_factor=2):
#         self.s3_client = boto3.client(
#             's3',
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             region_name=region_name
#         )
#         self.bucket_name = bucket_name
#         self.file_prefix = DEFAULT_FILE_PREFIX
#         self.time_format = DEFAULT_TIME_FORMAT
#         self.max_retries = max_retries
#         self.backoff_factor = backoff_factor
#         self.queue = Queue()
#         self.executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKER_COUNT)
#         self.shutdown_event = asyncio.Event()  # Event to signal shutdown
#
#         # Check if bucket exists or create it
#         if not self._bucket_exists(bucket_name):
#             try:
#                 self.s3_client.create_bucket(
#                     Bucket=bucket_name,
#                     CreateBucketConfiguration={'LocationConstraint': region_name}
#                 )
#                 logger.info(f"Bucket {bucket_name} created successfully.")
#             except ClientError as e:
#                 logger.error(f"Error creating bucket {bucket_name}: {e}")
#                 raise e
#
#     async def start_worker(self):
#         """Start the upload worker."""
#         await self._upload_worker()
#
#     def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
#         try:
#             # Add spans to the queue for batching
#             for span in spans:
#                 self.queue.put(span)
#
#             return SpanExportResult.SUCCESS
#         except Exception as e:
#             logger.error(f"Error exporting spans: {e}")
#             return SpanExportResult.FAILURE
#
#     async def _upload_worker(self):
#         """Worker function to batch spans and upload to S3 asynchronously."""
#         while not self.shutdown_event.is_set() or not self.queue.empty():
#             try:
#                 batch = []
#                 # Collect up to MAX_BATCH_SIZE spans from the queue
#                 while len(batch) < MAX_BATCH_SIZE and not self.queue.empty():
#                     span = self.queue.get()
#                     if span:
#                         batch.append(span)
#
#                 if batch:
#                     # Serialize the batch
#                     span_data_batch = self._serialize_spans(batch)
#                     # Upload asynchronously
#                     await asyncio.get_event_loop().run_in_executor(self.executor, self._upload_to_s3_with_retry,
#                                                                    span_data_batch)
#
#             except Exception as e:
#                 logger.error(f"Error in upload worker: {e}")
#
#     def _serialize_spans(self, spans: Sequence[ReadableSpan]) -> str:
#         try:
#             # Serialize spans to JSON or any other format you prefer
#             span_data_list = [span.to_json() for span in spans]
#             return "[" + ", ".join(span_data_list) + "]"
#         except Exception as e:
#             logger.error(f"Error serializing spans: {e}")
#             raise
#
#     def _upload_to_s3_with_retry(self, span_data_batch: str):
#         current_time = datetime.datetime.now().strftime(self.time_format)
#         file_name = f"{self.file_prefix}{current_time}.json"
#         attempt = 0
#
#         while attempt < self.max_retries:
#             try:
#                 self.s3_client.put_object(
#                     Bucket=self.bucket_name,
#                     Key=file_name,
#                     Body=span_data_batch
#                 )
#                 logger.info(f"Span batch uploaded to AWS S3 as {file_name}.")
#                 return
#
#             except EndpointConnectionError as e:
#                 logger.warning(
#                     f"Network connectivity error: {e}. Retrying in {self.backoff_factor ** attempt} seconds...")
#                 sleep_time = self.backoff_factor * (2 ** (attempt)) + random.uniform(0, 1)
#                 time.sleep(sleep_time)
#                 attempt += 1
#             except ClientError as e:
#                 error_code = e.response.get("Error", {}).get("Code", "")
#                 if error_code in ("RequestTimeout", "ThrottlingException", "InternalError", "ServiceUnavailable"):
#                     logger.warning(f"Retry {attempt}/{self.max_retries} failed due to network issue: {str(e)}")
#                 else:
#                     logger.error(f"Failed to upload trace data: {str(e)}")
#                     break
#
#                 sleep_time = self.backoff_factor * (2 ** (attempt)) + random.uniform(0, 1)
#                 logger.info(f"Waiting for {sleep_time:.2f} seconds before retrying...")
#                 time.sleep(sleep_time)
#                 attempt += 1
#             except (NoCredentialsError, ClientError) as e:
#                 logger.error(f"Error uploading spans to S3: {e}")
#                 raise
#             except Exception as e:
#                 logger.error(f"Unexpected error uploading spans to S3: {e}")
#                 raise
#
#         logger.error("Max retries exceeded. Failed to upload spans to S3.")
#         raise EndpointConnectionError(endpoint_url="S3 Upload Endpoint")
#
#     async def flush(self):
#         """Flush any remaining spans in the queue."""
#         # Signal the shutdown event to stop workers from pulling new tasks
#         self.shutdown_event.set()
#
#         # Wait until the queue is processed
#         while not self.queue.empty():
#             await asyncio.sleep(1)  # Wait for the queue to be processed
#
#         # Ensure all tasks complete before shutting down
#         self.executor.shutdown(wait=True)
#
#     def force_flush(self, timeout_millis: int = 30000) -> bool:
#         """Force flush the remaining spans within a timeout period."""
#         asyncio.run(self.flush())
#         return True
#
#     def shutdown(self) -> None:
#         asyncio.run(self.flush())  # Ensure all spans are uploaded
#         self.executor.shutdown(wait=True)
#
#     def _bucket_exists(self, bucket_name):
#         try:
#             self.s3_client.head_bucket(Bucket=bucket_name)
#             return True
#         except ClientError as e:
#             error_code = int(e.response['Error']['Code'])
#             if error_code == 404:
#                 return False
#             else:
#                 raise e
