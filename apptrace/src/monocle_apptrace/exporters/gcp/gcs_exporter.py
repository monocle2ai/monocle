import os
import datetime
import logging
import asyncio
from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden, GoogleCloudError, Conflict, TooManyRequests
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from typing import Sequence, Optional, Dict, List, Tuple
from monocle_apptrace.exporters.base_exporter import SpanExporterBase, format_trace_id_without_0x
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor
import json
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SDK_VERSION

logger = logging.getLogger(__name__)

HANDLE_TIMEOUT_SECONDS = 60


class GCSSpanExporter(SpanExporterBase):

    def __init__(
            self,
            bucket_name: Optional[str] = None,
            project_id: Optional[str] = None,
            location: Optional[str] = None,
            task_processor: Optional[ExportTaskProcessor] = None
    ):

        super().__init__()

        # Configuration constants
        DEFAULT_FILE_PREFIX = "monocle_trace_"
        DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
        DEFAULT_LOCATION = "US"

        self.max_batch_size = 500
        self.export_interval = 1

        # trace_id: {trace_id: (spans_list, creation_time, has_root_span)}
        self.trace_spans: Dict[int, Tuple[List[ReadableSpan], datetime.datetime, bool]] = {}

        if not bucket_name:
            bucket_name = os.getenv('MONOCLE_GCS_BUCKET_NAME')
            if not bucket_name:
                raise ValueError(
                    "GCS bucket name is not provided. Please provide bucket_name parameter "
                    "or set MONOCLE_GCS_BUCKET_NAME environment variable."
                )

        self.bucket_name = bucket_name
        self.project_id = project_id or os.getenv('MONOCLE_GCS_PROJECT_ID')
        self.location = location or os.getenv('MONOCLE_GCS_LOCATION', DEFAULT_LOCATION)
        self.file_prefix = os.getenv('MONOCLE_GCS_KEY_PREFIX', DEFAULT_FILE_PREFIX)
        self.time_format = DEFAULT_TIME_FORMAT

        try:
            if self.project_id:
                self.storage_client = storage.Client(project=self.project_id)
            else:
                self.storage_client = storage.Client()
                self.project_id = self.storage_client.project
                logger.info(f"Auto-detected GCP project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            logger.error(
                "Please ensure GOOGLE_APPLICATION_CREDENTIALS is set or "
                "Application Default Credentials are configured. "
                "See: https://cloud.google.com/docs/authentication/application-default-credentials"
            )
            raise

        self.bucket = self.storage_client.bucket(self.bucket_name)

        # Check if bucket exists or create it
        if not self.__bucket_exists(self.bucket_name):
            try:
                logger.info(f"Bucket {self.bucket_name} does not exist. Attempting to create it...")
                self.__create_bucket(self.bucket_name, self.location)
                logger.info(f"Bucket {self.bucket_name} created successfully in location {self.location}.")
            except Conflict:
                logger.warning(f"Bucket {self.bucket_name} was created by another process.")
            except Forbidden as e:
                logger.error(
                    f"Permission denied creating bucket {self.bucket_name}. "
                    f"Please ensure the service account has storage.buckets.create permission "
                    f"or create the bucket manually. Error: {e}"
                )
                raise PermissionError(
                    f"Cannot create bucket {self.bucket_name}. Please create it manually "
                    f"or grant storage.buckets.create permission."
                )
            except Exception as e:
                logger.error(f"Error creating bucket {self.bucket_name}: {e}")
                raise

        self.task_processor = task_processor
        if self.task_processor is not None:
            self.task_processor.start()

        logger.info(
            f"GCSSpanExporter initialized successfully. "
            f"Bucket: {self.bucket_name}, Project: {self.project_id}, Location: {self.location}"
        )

    def __bucket_exists(self, bucket_name: str) -> bool:

        try:
            exists = self.bucket.exists()
            if exists:
                logger.debug(f"Bucket {bucket_name} exists and is accessible.")
            else:
                logger.debug(f"Bucket {bucket_name} does not exist.")
            return exists
        except Forbidden as e:
            logger.error(f"Access to bucket {bucket_name} is forbidden")
            raise PermissionError(
                f"Access to bucket {bucket_name} is forbidden. "
                f"Please ensure the service account has storage.buckets.get permission."
            )
        except Exception as e:
            logger.error(f"Unexpected error when checking if bucket {bucket_name} exists: {e}")
            raise

    def __create_bucket(self, bucket_name: str, location: str) -> None:

        try:
            bucket = self.storage_client.create_bucket(
                bucket_or_name=bucket_name,
                location=location
            )
            logger.info(f"Created bucket {bucket_name} in location {location}")
        except Conflict as e:
            # 409 Conflict
            logger.error(
                f"Bucket name {bucket_name} is already taken globally. "
            )
            raise
        except Forbidden as e:
            logger.error(
                f"Permission denied to create bucket {bucket_name}. "
                f"Ensure service account has storage.buckets.create permission."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to create bucket {bucket_name}: {e}")
            raise

    def _cleanup_expired_traces(self) -> None:
        current_time = datetime.datetime.now()
        expired_trace_ids = []

        for trace_id, (spans, creation_time, _) in self.trace_spans.items():
            if (current_time - creation_time).total_seconds() > HANDLE_TIMEOUT_SECONDS:
                logger.warning(
                    f"Trace {format_trace_id_without_0x(trace_id)} has expired "
                    f"(timeout: {HANDLE_TIMEOUT_SECONDS}s). Uploading {len(spans)} spans."
                )
                expired_trace_ids.append(trace_id)

        for trace_id in expired_trace_ids:
            self._upload_trace(trace_id)

    def _add_spans_to_trace(self, trace_id: int, spans: List[ReadableSpan], has_root: bool = False) -> None:
        if trace_id in self.trace_spans:
            # Trace already exists, append spans
            existing_spans, creation_time, existing_root = self.trace_spans[trace_id]
            existing_spans.extend(spans)
            has_root = has_root or existing_root  # Update root flag
            self.trace_spans[trace_id] = (existing_spans, creation_time, has_root)
            logger.debug(
                f"Added {len(spans)} spans to existing trace {format_trace_id_without_0x(trace_id)}. "
                f"Total spans: {len(existing_spans)}, Has root: {has_root}"
            )
        else:
            # Create new trace entry
            self.trace_spans[trace_id] = (spans.copy(), datetime.datetime.now(), has_root)
            logger.debug(
                f"Created new trace buffer for {format_trace_id_without_0x(trace_id)} "
                f"with {len(spans)} spans. Has root: {has_root}"
            )

    def _upload_trace(self, trace_id: int) -> None:
        if trace_id not in self.trace_spans:
            logger.debug(f"Trace {format_trace_id_without_0x(trace_id)} not found in buffer.")
            return

        spans, _, _ = self.trace_spans[trace_id]
        if len(spans) == 0:
            logger.debug(f"Trace {format_trace_id_without_0x(trace_id)} has no spans. Skipping upload.")
            del self.trace_spans[trace_id]
            return

        serialized_data = self.__serialize_spans(spans)
        if serialized_data:
            try:
                self.__upload_to_gcs_with_trace_id(serialized_data, trace_id)
                logger.info(
                    f"Successfully uploaded trace {format_trace_id_without_0x(trace_id)} "
                    f"with {len(spans)} spans to GCS bucket {self.bucket_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to upload trace {format_trace_id_without_0x(trace_id)}: {e}",
                    exc_info=True
                )
        else:
            logger.warning(f"No valid data to upload for trace {format_trace_id_without_0x(trace_id)}")

        # Remove trace from buffer after upload attempt
        del self.trace_spans[trace_id]

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            # Run the asynchronous export logic in an event loop
            asyncio.run(self._export_async(spans))
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Error exporting spans to GCS: {e}", exc_info=True)
            return SpanExportResult.FAILURE

    async def _export_async(self, spans: Sequence[ReadableSpan]):
        try:
            # Cleanup expired traces first
            self._cleanup_expired_traces()

            # Group spans by trace_id
            spans_by_trace: Dict[int, List[ReadableSpan]] = {}
            root_span_traces = set()

            for span in spans:
                if self.skip_export(span):
                    logger.debug(f"Skipping export of non-Monocle span: {span.name}")
                    continue

                trace_id = span.context.trace_id
                if trace_id not in spans_by_trace:
                    spans_by_trace[trace_id] = []
                spans_by_trace[trace_id].append(span)

                if not span.parent:
                    root_span_traces.add(trace_id)
                    logger.debug(f"Found root span for trace {format_trace_id_without_0x(trace_id)}")

            # Add spans to their respective trace buffers
            for trace_id, trace_spans in spans_by_trace.items():
                has_root = trace_id in root_span_traces
                self._add_spans_to_trace(trace_id, trace_spans, has_root)

            # Upload complete traces
            for trace_id in root_span_traces:
                if self.task_processor is not None and callable(getattr(self.task_processor, 'queue_task', None)):
                    # Queue the upload task for async processing
                    if trace_id in self.trace_spans:
                        spans_to_upload, _, _ = self.trace_spans[trace_id]
                        serialized_data = self.__serialize_spans(spans_to_upload)
                        if serialized_data:
                            logger.debug(f"Queuing upload task for trace {format_trace_id_without_0x(trace_id)}")
                            self.task_processor.queue_task(
                                self.__upload_to_gcs_with_trace_id,
                                kwargs={'span_data_batch': serialized_data, 'trace_id': trace_id},
                                is_root_span=True
                            )
                        del self.trace_spans[trace_id]
                else:
                    self._upload_trace(trace_id)
        except Exception as e:
            logger.error(f"Error in _export_async: {e}", exc_info=True)

    def __serialize_spans(self, spans: Sequence[ReadableSpan]) -> str:
        try:
            valid_json_list = []
            for span in spans:
                try:
                    span_json = span.to_json(indent=0).replace("\n", "")  # make oneline json
                    valid_json_list.append(span_json)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON format in span data: {span.context.span_id}. Error: {e}"
                    )
                    continue
                except Exception as e:
                    logger.warning(f"Error serializing span {span.context.span_id}: {e}")
                    continue

            if not valid_json_list:
                logger.warning("No valid spans to serialize")
                return ""

            ndjson_data = "\n".join(valid_json_list) + "\n"
            logger.debug(f"Serialized {len(valid_json_list)} spans to NDJSON format")
            return ndjson_data
        except Exception as e:
            logger.error(f"Error serializing spans: {e}", exc_info=True)
            return ""

    @SpanExporterBase.retry_with_backoff(
        exceptions=(GoogleCloudError, TooManyRequests, ConnectionError)
    )
    def __upload_to_gcs_with_trace_id(self, span_data_batch: str, trace_id: int):
        try:
            # Generate filename with timestamp and trace ID
            current_time = datetime.datetime.now().strftime(self.time_format)
            file_name = f"{self.file_prefix}{current_time}_{format_trace_id_without_0x(trace_id)}.ndjson"

            blob = self.bucket.blob(file_name)
            blob.upload_from_string(
                data=span_data_batch,
                content_type='application/x-ndjson'
            )

            logger.debug(
                f"Trace {format_trace_id_without_0x(trace_id)} uploaded to "
                f"GCS bucket {self.bucket_name} as {file_name}."
            )
        except NotFound as e:
            logger.error(
                f"Bucket {self.bucket_name} not found. It may have been deleted. Error: {e}"
            )
            raise Exception(f"Bucket {self.bucket_name} does not exist") from e
        except Forbidden as e:
            logger.error(
                f"Permission denied uploading to bucket {self.bucket_name}. "
                f"Ensure service account has storage.objects.create permission. Error: {e}"
            )
            raise PermissionError(
                f"Cannot upload to bucket {self.bucket_name}. "
            ) from e
        except TooManyRequests as e:
            logger.warning(f"Rate limited by GCS. Will retry. Error: {e}")
            raise
        except GoogleCloudError as e:
            logger.error(f"GCS error uploading trace: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading to GCS: {e}", exc_info=True)
            raise

    async def force_flush(self, timeout_millis: int = 30000) -> bool:
        logger.info(f"Force flushing {len(self.trace_spans)} pending traces to GCS")
        trace_ids_to_upload = list(self.trace_spans.keys())
        for trace_id in trace_ids_to_upload:
            self._upload_trace(trace_id)
        logger.info("Force flush completed")
        return True

    def shutdown(self) -> None:
        logger.info("Shutting down GCSSpanExporter")

        trace_ids_to_upload = list(self.trace_spans.keys())
        if trace_ids_to_upload:
            logger.info(f"Uploading {len(trace_ids_to_upload)} remaining traces before shutdown")
            for trace_id in trace_ids_to_upload:
                self._upload_trace(trace_id)

        if hasattr(self, 'task_processor') and self.task_processor is not None:
            self.task_processor.stop()

        logger.info("GCSSpanExporter has been shut down successfully.")