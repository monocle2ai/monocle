import os
import time
import datetime
import logging
import asyncio
from typing import Sequence, Optional
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from monocle_apptrace.exporters.base_exporter import SpanExporterBase
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor
from opendal import Operator
from opendal.exceptions import PermissionDenied, ConfigInvalid, Unexpected

import json

logger = logging.getLogger(__name__)
class OpenDALS3Exporter(SpanExporterBase):
    def __init__(self, bucket_name=None, region_name=None, task_processor: Optional[ExportTaskProcessor] = None):
        super().__init__()
        DEFAULT_FILE_PREFIX = "monocle_trace_"
        DEFAULT_TIME_FORMAT = "%Y-%m-%d__%H.%M.%S"
        self.max_batch_size = 500
        self.export_interval = 1
        self.file_prefix = DEFAULT_FILE_PREFIX
        self.time_format = DEFAULT_TIME_FORMAT
        self.export_queue = []
        self.last_export_time = time.time()
        self.bucket_name = bucket_name or os.getenv("MONOCLE_S3_BUCKET_NAME", "default-bucket")

        # Initialize OpenDAL S3 operator
        self.op = Operator(
            "s3",
            root = "/",
            region=os.getenv("AWS_REGION", region_name),
            bucket=self.bucket_name,
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        
        self.task_processor = task_processor
        if self.task_processor is not None:
            self.task_processor.start()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Synchronous export method that internally handles async logic."""
        try:
            # Run the asynchronous export logic in an event loop
            asyncio.run(self.__export_async(spans))
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Error exporting spans: {e}")
            return SpanExportResult.FAILURE

    async def __export_async(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            # Add spans to the export queue
            for span in spans:
                self.export_queue.append(span)
                if len(self.export_queue) >= self.max_batch_size:
                    await self.__export_spans()

            # Check if it's time to force a flush
            current_time = time.time()
            if current_time - self.last_export_time >= self.export_interval:
                await self.__export_spans()
                self.last_export_time = current_time

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
            return "\n".join(valid_json_list) + "\n"
        except Exception as e:
            logger.warning(f"Error serializing spans: {e}")

    async def __export_spans(self):
        if not self.export_queue:
            return
        # Take a batch of spans from the queue
        batch_to_export = self.export_queue[:self.max_batch_size]
        serialized_data = self.__serialize_spans(batch_to_export)
        self.export_queue = self.export_queue[self.max_batch_size:]
        
        # Calculate is_root_span by checking if any span has no parent
        is_root_span = any(not span.parent for span in batch_to_export)
        
        if self.task_processor is not None and callable(getattr(self.task_processor, 'queue_task', None)):
            self.task_processor.queue_task(self.__upload_to_s3, serialized_data, is_root_span)
        else:
            try:
                self.__upload_to_s3(serialized_data, is_root_span)
            except Exception as e:
                logger.error(f"Failed to upload span batch: {e}")

    @SpanExporterBase.retry_with_backoff(exceptions=(Unexpected))
    def __upload_to_s3(self, span_data_batch: str, is_root_span: bool = False):
        current_time = datetime.datetime.now().strftime(self.time_format)
        file_name = f"{self.file_prefix}{current_time}.ndjson"
        try:
            # Attempt to write the span data batch to S3
            self.op.write(file_name, span_data_batch.encode("utf-8"))
            logger.info(f"Span batch uploaded to S3 as {file_name}. Is root span: {is_root_span}")

        except PermissionDenied as e:
            # S3 bucket is forbidden.
            logger.error(f"Access to bucket {self.bucket_name} is forbidden (403).")
            raise PermissionError(f"Access to bucket {self.bucket_name} is forbidden.")

        except ConfigInvalid as e:
            # Bucket does not exist.
            if "404" in str(e):
                logger.error("Bucket does not exist. Please check the bucket name and region.")
                raise Exception(f"Bucket does not exist. Error: {e}")
            else:
                logger.error(f"Unexpected error when accessing bucket {self.bucket_name}: {e}")
                raise e


    async def force_flush(self, timeout_millis: int = 30000) -> bool:
        await self.__export_spans()
        return True

    def shutdown(self) -> None:
        if hasattr(self, 'task_processor') and self.task_processor is not None:
            self.task_processor.stop()
        logger.info("S3SpanExporter has been shut down.")
