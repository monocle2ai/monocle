#pylint: disable=consider-using-with

from os import linesep, path
from io import TextIOWrapper
from datetime import datetime
from typing import Optional, Callable, Sequence
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import SERVICE_NAME

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
    ):
        self.out_handle:TextIOWrapper = None
        self.formatter = formatter
        self.service_name = service_name
        self.output_path = out_path
        self.file_prefix = file_prefix
        self.time_format = time_format

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
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
        self.reset_handle()
