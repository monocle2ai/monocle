import os
import time
import datetime
import logging
import asyncio
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from typing import Sequence, Optional
from opendal import Operator
from monocle_apptrace.exporters.base_exporter import SpanExporterBase
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor
from opendal.exceptions import Unexpected, PermissionDenied, NotFound
import json

logger = logging.getLogger(__name__)

class OpenDALAzureExporter(SpanExporterBase):
    def __init__(self, connection_string=None, container_name=None, task_processor: Optional[ExportTaskProcessor] = None):
        super().__init__()
        DEFAULT_FILE_PREFIX = "monocle_trace_"
        DEFAULT_TIME_FORMAT = "%Y-%m-%d_%H.%M.%S"
        self.max_batch_size = 500
        self.export_interval = 1
        self.container_name = container_name

        # Default values
        self.file_prefix = DEFAULT_FILE_PREFIX
        self.time_format = DEFAULT_TIME_FORMAT
        self.export_queue = []  # Add this line to initialize export_queue
        self.last_export_time = time.time()  # Add this line to initialize last_export_time

        # Validate input
        if not connection_string:
            connection_string = os.getenv('MONOCLE_BLOB_CONNECTION_STRING')
            if not connection_string:
                raise ValueError("Azure Storage connection string is not provided or set in environment variables.")

        if not container_name:
            container_name = os.getenv('MONOCLE_BLOB_CONTAINER_NAME', 'default-container')
        endpoint, account_name , account_key = self.parse_connection_string(connection_string)

        if not account_name or not account_key:
            raise ValueError("AccountName or AccountKey missing in the connection string.")

        try:
            # Initialize OpenDAL operator with explicit credentials
            self.operator = Operator(
                "azblob",
                endpoint=endpoint,
                account_name=account_name,
                account_key=account_key,
                container=container_name
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenDAL operator: {e}")

        self.task_processor = task_processor
        if self.task_processor is not None:
            self.task_processor.start()

    def parse_connection_string(self,connection_string):
        connection_params = dict(item.split('=', 1) for item in connection_string.split(';') if '=' in item)

        account_name = connection_params.get('AccountName')
        account_key = connection_params.get('AccountKey')
        endpoint_suffix = connection_params.get('EndpointSuffix')

        if not all([account_name, account_key, endpoint_suffix]):
            raise ValueError("Invalid connection string. Ensure it contains AccountName, AccountKey, and EndpointSuffix.")

        endpoint = f"https://{account_name}.blob.{endpoint_suffix}"
        return endpoint, account_name, account_key


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
            self.task_processor.queue_task(self.__upload_to_opendal, serialized_data, is_root_span)
        else:
            try:
                self.__upload_to_opendal(serialized_data, is_root_span)
            except Exception as e:
                logger.error(f"Failed to upload span batch: {e}")

    @SpanExporterBase.retry_with_backoff(exceptions=(Unexpected,))
    def __upload_to_opendal(self, span_data_batch: str, is_root_span: bool = False):
        current_time = datetime.datetime.now().strftime(self.time_format)
        file_name = f"{self.file_prefix}{current_time}.ndjson"

        try:
            self.operator.write(file_name, span_data_batch.encode('utf-8'))
            logger.info(f"Span batch uploaded to Azure Blob Storage as {file_name}. Is root span: {is_root_span}")
        except PermissionDenied as e:
            # Azure Container is forbidden.
            logger.error(f"Access to container {self.container_name} is forbidden (403).")
            raise PermissionError(f"Access to container {self.container_name} is forbidden.")

        except NotFound as e:
            # Container does not exist.
            if "404" in str(e):
                logger.error("Container does not exist. Please check the container name.")
                raise Exception(f"Container does not exist. Error: {e}")
            else:
                logger.error(f"Unexpected NotFound error when accessing container {self.container_name}: {e}")
                raise e

    async def force_flush(self, timeout_millis: int = 30000) -> bool:
        await self.__export_spans()
        return True

    def shutdown(self) -> None:
        if hasattr(self, 'task_processor') and self.task_processor is not None:
            self.task_processor.stop()
        logger.info("OpenDALAzureExporter has been shut down.")
