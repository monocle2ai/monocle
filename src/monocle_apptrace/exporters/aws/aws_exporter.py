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
    def __init__(self, bucket_name=None, region_name="us-east-1"):
        # Use environment variables if credentials are not provided
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region_name,
        )
        self.bucket_name = bucket_name or os.getenv('AWS_S3_BUCKET_NAME', 'default-bucket')
        self.file_prefix = DEFAULT_FILE_PREFIX
        self.time_format = DEFAULT_TIME_FORMAT
        self.export_queue = []
        self.last_export_time = time.time()

        # Check if bucket exists or create it
        if not self._bucket_exists(self.bucket_name):
            try:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region_name}
                )
                logger.info(f"Bucket {self.bucket_name} created successfully.")
            except ClientError as e:
                logger.error(f"Error creating bucket {self.bucket_name}: {e}")
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

