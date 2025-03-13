import os
import time
import random
import datetime
import logging
import asyncio
import boto3
from botocore.exceptions import ClientError
from botocore.exceptions import (
    BotoCoreError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from monocle_apptrace.exporters.base_exporter import SpanExporterBase
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor
from typing import Sequence, Optional
import json
logger = logging.getLogger(__name__)

class S3SpanExporter(SpanExporterBase):
    def __init__(self, bucket_name=None, region_name=None, task_processor: Optional[ExportTaskProcessor] = None):
        super().__init__()
        # Use environment variables if credentials are not provided
        DEFAULT_FILE_PREFIX = "monocle_trace_"
        DEFAULT_TIME_FORMAT = "%Y-%m-%d__%H.%M.%S"
        self.max_batch_size = 500
        self.export_interval = 1
        if(os.getenv('MONOCLE_AWS_ACCESS_KEY_ID') and os.getenv('MONOCLE_AWS_SECRET_ACCESS_KEY')):
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('MONOCLE_AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('MONOCLE_AWS_SECRET_ACCESS_KEY'),
                region_name=region_name,
            )
        else:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=region_name,
            )
        self.bucket_name = bucket_name or os.getenv('MONOCLE_S3_BUCKET_NAME','default-bucket')
        self.file_prefix = os.getenv('MONOCLE_S3_KEY_PREFIX', DEFAULT_FILE_PREFIX)
        self.time_format = DEFAULT_TIME_FORMAT
        self.export_queue = []
        self.last_export_time = time.time()
        self.task_processor = task_processor
        if self.task_processor is not None:
            self.task_processor.start()

        # Check if bucket exists or create it
        if not self.__bucket_exists(self.bucket_name):
            try:
                self.s3_client.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region_name}
                 )
                logger.info(f"Bucket {self.bucket_name} created successfully.")
            except ClientError as e:
                logger.error(f"Error creating bucket {self.bucket_name}: {e}")
                raise e

    def __bucket_exists(self, bucket_name):
        try:
            # Check if the bucket exists by calling head_bucket
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket not found
                logger.error(f"Bucket {bucket_name} does not exist (404).")
                return False
            elif error_code == '403':
                # Permission denied
                logger.error(f"Access to bucket {bucket_name} is forbidden (403).")
                raise PermissionError(f"Access to bucket {bucket_name} is forbidden.")
            elif error_code == '400':
                # Bad request or malformed input
                logger.error(f"Bad request for bucket {bucket_name} (400).")
                raise ValueError(f"Bad request for bucket {bucket_name}.")
            else:
                # Other client errors
                logger.error(f"Unexpected error when accessing bucket {bucket_name}: {e}")
                raise e
        except TypeError as e:
            # Handle TypeError separately
            logger.error(f"Type error while checking bucket existence: {e}")
            raise e

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Synchronous export method that internally handles async logic."""
        try:
            # Run the asynchronous export logic in an event loop
            logger.info(f"Exporting {len(spans)} spans to S3.")
            asyncio.run(self.__export_async(spans))
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Error exporting spans: {e}")
            return SpanExportResult.FAILURE

    async def __export_async(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            logger.info(f"__export_async {len(spans)} spans to S3.")
            # Add spans to the export queue
            for span in spans:
                self.export_queue.append(span)
                # If the queue reaches MAX_BATCH_SIZE, export the spans
                if len(self.export_queue) >= self.max_batch_size:
                    await self.__export_spans()

            # Check if it's time to force a flush
            current_time = time.time()
            if current_time - self.last_export_time >= self.export_interval:
                await self.__export_spans()  # Export spans if time interval has passed
                self.last_export_time = current_time  # Reset the last export time

            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Error exporting spans: {e}")
            return SpanExportResult.FAILURE

    def __serialize_spans(self, spans: Sequence[ReadableSpan]) -> str:
        try:
            # Serialize spans to JSON or any other format you prefer
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

        # Take a batch of spans from the queue
        batch_to_export = self.export_queue[:self.max_batch_size]
        serialized_data = self.__serialize_spans(batch_to_export)
        self.export_queue = self.export_queue[self.max_batch_size:]
        # to calculate is_root_span loop over each span in batch_to_export and check if parent id is none or null
        is_root_span = any(not span.parent for span in batch_to_export)
        logger.info(f"Exporting {len(batch_to_export)} spans to S3 is_root_span : {is_root_span}.")
        if self.task_processor is not None and callable(getattr(self.task_processor, 'queue_task', None)):
            self.task_processor.queue_task(self.__upload_to_s3, serialized_data, is_root_span)
        else:
            try:
                self.__upload_to_s3(serialized_data)
            except Exception as e:
                logger.error(f"Failed to upload span batch: {e}")

    @SpanExporterBase.retry_with_backoff(exceptions=(EndpointConnectionError, ConnectionClosedError, ReadTimeoutError, ConnectTimeoutError))
    def __upload_to_s3(self, span_data_batch: str):
        current_time = datetime.datetime.now().strftime(self.time_format)
        prefix = self.file_prefix + os.environ.get('MONOCLE_S3_KEY_PREFIX_CURRENT', '')
        file_name = f"{prefix}{current_time}.ndjson"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=file_name,
            Body=span_data_batch
        )
        logger.info(f"Span batch uploaded to AWS S3 as {file_name}.")

    async def force_flush(self, timeout_millis: int = 30000) -> bool:
        await self.__export_spans()  # Export any remaining spans in the queue
        return True

    def shutdown(self) -> None:
        if hasattr(self, 'task_processor') and self.task_processor is not None:
            self.task_processor.stop()
        logger.info("S3SpanExporter has been shut down.")
