#pylint: disable=consider-using-with

from os import linesep, path
from io import TextIOWrapper
from datetime import datetime
from typing import Optional, Callable, Sequence
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import SERVICE_NAME
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor

DEFAULT_FILE_PREFIX:str = "monocle_trace_"
DEFAULT_TIME_FORMAT:str = "%Y-%m-%d_%H.%M.%S"

class FileSpanExporter(SpanExporter):
    current_trace_id: int = None
    current_file_path: str = None

    def __init__(
        self,
        service_name: Optional[str] = None,
        out_path:str = ".",
        file_prefix = DEFAULT_FILE_PREFIX,
        time_format = DEFAULT_TIME_FORMAT,
        formatter: Callable[
            [ReadableSpan], str
        ] = lambda span: span.to_json()
        + linesep,
        task_processor: Optional[ExportTaskProcessor] = None
    ):
        self.out_handle:TextIOWrapper = None
        self.formatter = formatter
        self.service_name = service_name
        self.output_path = out_path
        self.file_prefix = file_prefix
        self.time_format = time_format
        self.task_processor = task_processor
        if self.task_processor is not None:
            self.task_processor.start()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self.task_processor is not None and callable(getattr(self.task_processor, 'queue_task', None)):
            # Check if any span is a root span (no parent)
            is_root_span = any(not span.parent for span in spans)
            self.task_processor.queue_task(self._process_spans, spans, is_root_span)
            return SpanExportResult.SUCCESS
        else:
            return self._process_spans(spans)

    def _process_spans(self, spans: Sequence[ReadableSpan], is_root_span: bool = False) -> SpanExportResult:
        for span in spans:
            if span.context.trace_id != self.current_trace_id:
                self.rotate_file(span.resource.attributes[SERVICE_NAME],
                                span.context.trace_id)
            self.out_handle.write(self.formatter(span))
        self.out_handle.flush()
        return SpanExportResult.SUCCESS

    def rotate_file(self, trace_name:str, trace_id:int) -> None:
        self.reset_handle()
        self.current_file_path = path.join(self.output_path,
                        self.file_prefix + trace_name + "_" + hex(trace_id) + "_"
                        + datetime.now().strftime(self.time_format) + ".json")
        self.out_handle = open(self.current_file_path, "w", encoding='UTF-8')
        self.current_trace_id = trace_id

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self.out_handle.flush()
        return True

    def reset_handle(self) -> None:
        if self.out_handle is not None:
            self.out_handle.close()
            self.out_handle = None

    def shutdown(self) -> None:
        if hasattr(self, 'task_processor') and self.task_processor is not None:
            self.task_processor.stop()
        self.reset_handle()
