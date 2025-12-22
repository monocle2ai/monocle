import os
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
from monocle_apptrace.exporters.base_exporter import SpanExporterBase, format_trace_id_without_0x
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor
from typing import Sequence, Optional, Dict, List, Tuple
import json
logger = logging.getLogger(__name__)

HANDLE_TIMEOUT_SECONDS = 60

class S3SpanExporter(SpanExporterBase):
    def __init__(self, bucket_name=None, region_name=None, task_processor: Optional[ExportTaskProcessor] = None):
        super().__init__()
        # Use environment variables if credentials are not provided
        DEFAULT_FILE_PREFIX = "monocle_trace_"
        DEFAULT_TIME_FORMAT = "%Y-%m-%d__%H.%M.%S"
        self.max_batch_size = 500
        self.export_interval = 1
        # Dictionary to store spans by trace_id: {trace_id: (spans_list, creation_time, has_root_span)}
        self.trace_spans: Dict[int, Tuple[List[ReadableSpan], datetime.datetime, bool]] = {}
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
        """Upload a specific trace to S3 and remove it from the buffer."""
        if trace_id not in self.trace_spans:
            return
        
        spans, _, _ = self.trace_spans[trace_id]
        if len(spans) == 0:
            del self.trace_spans[trace_id]
            return
        
        serialized_data = self.__serialize_spans(spans)
        if serialized_data:
            try:
                self.__upload_to_s3_with_trace_id(span_data_batch=serialized_data, trace_id=trace_id)
            except Exception as e:
                logger.error(f"Failed to upload trace {format_trace_id_without_0x(trace_id)}: {e}")
        
        del self.trace_spans[trace_id]

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
            
            # Cleanup expired traces first
            self._cleanup_expired_traces()
            
            # Group spans by trace_id
            spans_by_trace = {}
            root_span_traces = set()
            
            for span in spans:
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
                                self.__upload_to_s3_with_trace_id,
                                kwargs={'span_data_batch': serialized_data, 'trace_id': trace_id},
                                is_root_span=True
                            )
                        del self.trace_spans[trace_id]
                else:
                    self._upload_trace(trace_id)

            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Error exporting spans: {e}")
            return SpanExportResult.FAILURE

    def __serialize_spans(self, spans: Sequence[ReadableSpan]) -> str:
        try:
            # Serialize spans to JSON or any other format you prefer
            valid_json_list = []
            for span in spans:
                if self.skip_export(span):
                    continue
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

    @SpanExporterBase.retry_with_backoff(exceptions=(EndpointConnectionError, ConnectionClosedError, ReadTimeoutError, ConnectTimeoutError))
    def __upload_to_s3_with_trace_id(self, span_data_batch: str, trace_id: int) -> None:
        """Upload spans for a specific trace to S3 with trace ID in filename."""
        current_time = datetime.datetime.now().strftime(self.time_format)
        prefix = self.file_prefix + os.environ.get('MONOCLE_S3_KEY_PREFIX_CURRENT', '')
        file_name = f"{prefix}{current_time}_{format_trace_id_without_0x(trace_id)}.ndjson"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=file_name,
            Body=span_data_batch
        )
        logger.debug(f"Trace {format_trace_id_without_0x(trace_id)} uploaded to AWS S3 as {file_name}.")

    async def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush all pending traces to S3."""
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
        logger.info("S3SpanExporter has been shut down.")
