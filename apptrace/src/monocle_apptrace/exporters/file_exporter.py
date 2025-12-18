#pylint: disable=consider-using-with

from os import linesep, path
from io import TextIOWrapper
from datetime import datetime
import os
from typing import Optional, Callable, Sequence, Dict, Tuple
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import SERVICE_NAME
from monocle_apptrace.exporters.base_exporter import SpanExporterBase, format_trace_id_without_0x
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor

DEFAULT_FILE_PREFIX:str = "monocle_trace_"
DEFAULT_TIME_FORMAT:str = "%Y-%m-%d_%H.%M.%S"
HANDLE_TIMEOUT_SECONDS: int = 60  # 1 minute timeout
DEFAULT_TRACE_FOLDER = ".monocle"

class FileSpanExporter(SpanExporterBase):
    def __init__(
        self,
        service_name: Optional[str] = None,
        out_path:str = path.join(".", DEFAULT_TRACE_FOLDER),
        file_prefix = DEFAULT_FILE_PREFIX,
        time_format = DEFAULT_TIME_FORMAT,
        formatter: Callable[
            [ReadableSpan], str
        ] = lambda span: span.to_json(indent = 4)
        + linesep,
        task_processor: Optional[ExportTaskProcessor] = None
    ):
        super().__init__()
        # Dictionary to store file handles: {trace_id: (file_handle, file_path, creation_time, first_span)}
        self.file_handles: Dict[int, Tuple[TextIOWrapper, str, datetime, bool]] = {}
        self.formatter = formatter
        self.service_name = service_name
        self.output_path = os.getenv("MONOCLE_TRACE_OUTPUT_PATH", out_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.file_prefix = file_prefix
        self.time_format = time_format
        self.task_processor = task_processor
        self.is_first_span_in_file = True  # Track if this is the first span in the current file
        if self.task_processor is not None:
            self.task_processor.start()
        self.last_file_processed:str = None
        self.last_trace_id = None

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        is_root_span = any(not span.parent for span in spans)
        if self.task_processor is not None and callable(getattr(self.task_processor, 'queue_task', None)):
            # Check if any span is a root span (no parent)
            self.task_processor.queue_task(
                self._process_spans,
                kwargs={'spans': spans, 'is_root_span': is_root_span},
                is_root_span=is_root_span
            )
            return SpanExportResult.SUCCESS
        else:
            return self._process_spans(spans, is_root_span=is_root_span)

    def set_service_name(self, service_name: str) -> None:
        self.service_name = service_name

    def _cleanup_expired_handles(self) -> None:
        """Close and remove file handles that have exceeded the timeout."""
        current_time = datetime.now()
        expired_trace_ids = []
        
        for trace_id, (handle, file_path, creation_time, _) in self.file_handles.items():
            if (current_time - creation_time).total_seconds() > HANDLE_TIMEOUT_SECONDS:
                expired_trace_ids.append(trace_id)
        
        for trace_id in expired_trace_ids:
            self._close_trace_handle(trace_id)

    def _get_or_create_handle(self, trace_id: int, service_name: str) -> Tuple[TextIOWrapper, str, bool]:
        """Get existing handle or create new one for the trace_id."""
        self._cleanup_expired_handles()
        
        if trace_id in self.file_handles:
            handle, file_path, creation_time, first_span = self.file_handles[trace_id]
            return handle, file_path, first_span
        
        # Create new handle
        file_path = path.join(self.output_path,
                             self.file_prefix + service_name + "_" + format_trace_id_without_0x(trace_id) + "_"
                             + datetime.now().strftime(self.time_format) + ".json")
        
        try:
            handle = open(file_path, "w", encoding='UTF-8')
            handle.write("[")
            self.file_handles[trace_id] = (handle, file_path, datetime.now(), True)
            return handle, file_path, True
        except Exception as e:
            print(f"Error creating file {file_path}: {e}")
            return None, file_path, True

    def _close_trace_handle(self, trace_id: int) -> None:
        """Close and remove a specific trace handle."""
        if trace_id in self.file_handles:
            handle, file_path, creation_time, _ = self.file_handles[trace_id]
            try:
                if handle is not None:
                    handle.write("]")
                    handle.close()
            except Exception as e:
                print(f"Error closing file {file_path}: {e}")
            finally:
                del self.file_handles[trace_id]
                self.last_file_processed = file_path
                self.last_trace_id = trace_id

    def _mark_span_written(self, trace_id: int) -> None:
        """Mark that a span has been written for this trace (no longer first span)."""
        if trace_id in self.file_handles:
            handle, file_path, creation_time, _ = self.file_handles[trace_id]
            self.file_handles[trace_id] = (handle, file_path, creation_time, False)

    def _process_spans(self, spans: Sequence[ReadableSpan], is_root_span: bool = False) -> SpanExportResult:
        # Group spans by trace_id for efficient processing
        spans_by_trace = {}
        root_span_traces = set()
        
        for span in spans:
            if self.skip_export(span):
                continue
            
            trace_id = span.context.trace_id
            if trace_id not in spans_by_trace:
                spans_by_trace[trace_id] = []
            spans_by_trace[trace_id].append(span)
            
            # Check if this span is a root span
            if not span.parent:
                root_span_traces.add(trace_id)
        
        # Process spans for each trace
        for trace_id, trace_spans in spans_by_trace.items():
            if self.service_name is not None:
                service_name = self.service_name
            else:
                service_name = trace_spans[0].resource.attributes.get(SERVICE_NAME, "unknown")
            handle, file_path, is_first_span = self._get_or_create_handle(trace_id, service_name)
            
            if handle is None:
                continue
            
            for span in trace_spans:
                if not is_first_span:
                    try:
                        handle.write(",")
                    except Exception as e:
                        print(f"Error writing comma to file {file_path} for span {span.context.span_id}: {e}")
                        continue
                
                try:
                    handle.write(self.formatter(span))
                    if is_first_span:
                        self._mark_span_written(trace_id)
                        is_first_span = False
                except Exception as e:
                    print(f"Error formatting span {span.context.span_id}: {e}")
                    continue
        
        # Close handles for traces with root spans
        for trace_id in root_span_traces:
            self._close_trace_handle(trace_id)
        
        # Flush remaining handles
        for trace_id, (handle, file_path, _, _) in self.file_handles.items():
            if trace_id not in root_span_traces:
                try:
                    if handle is not None:
                        handle.flush()
                except Exception as e:
                    print(f"Error flushing file {file_path}: {e}")
        
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush all open file handles."""
        for trace_id, (handle, file_path, _, _) in self.file_handles.items():
            try:
                if handle is not None:
                    handle.flush()
            except Exception as e:
                print(f"Error flushing file {file_path}: {e}")
        return True

    def shutdown(self) -> None:
        """Close all file handles and stop task processor."""
        if hasattr(self, 'task_processor') and self.task_processor is not None:
            self.task_processor.stop()
        
        # Close all remaining file handles
        trace_ids_to_close = list(self.file_handles.keys())
        for trace_id in trace_ids_to_close:
            self._close_trace_handle(trace_id)
