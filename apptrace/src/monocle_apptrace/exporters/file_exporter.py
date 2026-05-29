#pylint: disable=consider-using-with

import json
from os import linesep, path
from io import TextIOWrapper
from datetime import datetime
import os
from typing import Optional, Callable, Sequence, Dict, Tuple
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import SERVICE_NAME
from monocle_apptrace.exporters.base_exporter import SpanExporterBase, format_trace_id_without_0x, serialize_span
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor

DEFAULT_FILE_PREFIX:str = "monocle_trace_"
DEFAULT_TIME_FORMAT:str = "%Y-%m-%d_%H.%M.%S"
HANDLE_TIMEOUT_SECONDS: int = 60  # 1 minute timeout for individual trace handles
SESSION_HANDLE_TIMEOUT_SECONDS: int = 600  # 10 minute timeout for session handles (turns can be minutes apart)
DEFAULT_TRACE_FOLDER = ".monocle"
# Sentinel so we can tell "caller passed nothing" apart from "caller passed default".
_UNSET = object()

class FileSpanExporter(SpanExporterBase):
    def __init__(
        self,
        service_name: Optional[str] = None,
        out_path:str = path.join(".", DEFAULT_TRACE_FOLDER),
        file_prefix = _UNSET,
        time_format = DEFAULT_TIME_FORMAT,
        formatter: Callable[
            [ReadableSpan], str
        ] = lambda span: json.dumps(serialize_span(span), indent=4)
        + linesep,
        task_processor: Optional[ExportTaskProcessor] = None
    ):
        super().__init__()
        # Dictionary to store file handles: {trace_id: (file_handle, file_path, creation_time, first_span)}
        self.file_handles: Dict[int, Tuple[TextIOWrapper, str, datetime, bool]] = {}
        # Session-based handles for multi-turn stitching: {session_id: (file_handle, file_path, creation_time, first_span)}
        # When a span carries scope.agentic.session, all traces in that session share one file.
        self._session_handles: Dict[str, Tuple[TextIOWrapper, str, datetime, bool]] = {}
        self.formatter = formatter
        self.service_name = service_name
        self.output_path = os.getenv("MONOCLE_TRACE_OUTPUT_PATH", out_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # Precedence: explicit constructor arg > MONOCLE_FILE_PREFIX env var > default.
        if file_prefix is _UNSET:
            self.file_prefix = os.getenv("MONOCLE_FILE_PREFIX", DEFAULT_FILE_PREFIX)
        else:
            self.file_prefix = file_prefix
        self.time_format = time_format
        self.task_processor = task_processor
        self.is_first_span_in_file = True  # Track if this is the first span in the current file
        if self.task_processor is not None:
            self.task_processor.start()
        self.last_file_processed:str = None
        self.last_trace_id = None
        self._root_span_seen: set = set()  # traces where root arrived but child hasn't yet

    @staticmethod
    def _is_root_span(span: ReadableSpan) -> bool:
        """Return True if the span has no parent (i.e. it is a root span)."""
        return (not span.parent) or (span.attributes.get("span.type") == "workflow")

    @staticmethod
    def _get_session_id(spans) -> Optional[str]:
        """Return the agentic session ID shared by these spans, or None."""
        for span in spans:
            session_id = span.attributes.get("scope.agentic.session")
            if session_id:
                return str(session_id)
        return None


    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        is_root_span = any(FileSpanExporter._is_root_span(span) for span in spans)
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
        expired_sessions = []
        for trace_id, (handle, file_path, creation_time, _) in self.file_handles.items():
            if (current_time - creation_time).total_seconds() > HANDLE_TIMEOUT_SECONDS:
                expired_trace_ids.append(trace_id)

        for session_id, (handle, file_path, creation_time, _) in self._session_handles.items():
            if (current_time - creation_time).total_seconds() > SESSION_HANDLE_TIMEOUT_SECONDS:
                expired_sessions.append(session_id)

        for trace_id in expired_trace_ids:
            self._close_trace_handle(trace_id)
        for session_id in expired_sessions:
            self._close_session_handle(session_id)

    def _close_session_handle(self, session_id: str) -> None:
        """Close and remove a session-level file handle."""
        if session_id in self._session_handles:
            handle, file_path, creation_time, _ = self._session_handles[session_id]
            try:
                if handle is not None:
                    handle.write("]")
                    handle.close()
            except Exception as e:
                print(f"Error closing session file {file_path}: {e}")
            finally:
                del self._session_handles[session_id]
                self.last_file_processed = file_path

    def _get_or_create_handle(self, trace_id: int, service_name: str, session_id: Optional[str] = None) -> Tuple[TextIOWrapper, str, bool]:
        """Get existing handle or create new one for the trace_id.

        If session_id is provided, reuse an existing session-level file so that
        all turns of the same conversation land in one file.  The internal
        trace_id-keyed bookkeeping is otherwise unchanged.
        """
        self._cleanup_expired_handles()

        if session_id:
            if session_id in self._session_handles:
                handle, file_path, creation_time, first_span = self._session_handles[session_id]
                self.file_handles[trace_id] = (handle, file_path, creation_time, first_span)
                return handle, file_path, first_span

        if trace_id in self.file_handles:
            handle, file_path, creation_time, first_span = self.file_handles[trace_id]
            return handle, file_path, first_span
        
        # Create new handle
        file_id = session_id.replace("/", "_").replace("\\", "_")[:64] if session_id \
            else format_trace_id_without_0x(trace_id)
        file_path = path.join(self.output_path,
                             self.file_prefix + service_name + "_" + file_id + "_"
                             + datetime.now().strftime(self.time_format) + ".json")
        
        try:
            handle = open(file_path, "w", encoding='UTF-8')
            handle.write("[")
            entry = (handle, file_path, datetime.now(), True)
            self.file_handles[trace_id] = entry
            if session_id:
                self._session_handles[session_id] = entry
            return handle, file_path, True
        except Exception as e:
            print(f"Error creating file {file_path}: {e}")
            return None, file_path, True

    def _close_trace_handle(self, trace_id: int) -> None:
        """Close and remove a specific trace handle.

        For session-backed files the underlying file stays open (managed by
        _session_handles); we only remove the trace_id alias.
        """
        if trace_id in self.file_handles:
            handle, file_path, creation_time, _ = self.file_handles[trace_id]
            # Check whether this handle belongs to an open session file.
            # If so, leave the file open — the session handle owns it.
            is_session_file = any(
                sh is handle
                for (sh, _, _, _) in self._session_handles.values()
            )
            try:
                if handle is not None and not is_session_file:
                    handle.write("]")
                    handle.close()
            except Exception as e:
                print(f"Error closing file {file_path}: {e}")
            finally:
                del self.file_handles[trace_id]
                self.last_file_processed = file_path
                self.last_trace_id = trace_id

    def _is_first_span(self, trace_id: int) -> bool:
        """Return True if no spans have been written yet for this trace (still the first span)."""
        if trace_id in self.file_handles:
            return self.file_handles[trace_id][3]
        return True

    def _mark_span_written(self, trace_id: int) -> None:
        """Mark that a span has been written for this trace (no longer first span)."""
        if trace_id in self.file_handles:
            handle, file_path, creation_time, _ = self.file_handles[trace_id]
            entry = (handle, file_path, creation_time, False)
            self.file_handles[trace_id] = entry
            for session_id, sh_entry in self._session_handles.items():
                if sh_entry[0] is handle:
                    self._session_handles[session_id] = entry
                    break

    def _process_spans(self, spans: Sequence[ReadableSpan], is_root_span: bool = False) -> SpanExportResult:
        spans_by_trace: Dict[int, list] = {}
        root_span_traces: set = set()

        for span in spans:
            if self.skip_export(span):
                continue
            
            trace_id = span.context.trace_id
            if trace_id not in spans_by_trace:
                spans_by_trace[trace_id] = []
            spans_by_trace[trace_id].append(span)
            
            # Check if this span is a root span
            if FileSpanExporter._is_root_span(span):
                root_span_traces.add(trace_id)
        
        # Process spans for each trace
        for trace_id, trace_spans in spans_by_trace.items():
            if self.service_name is not None:
                service_name = self.service_name
            else:
                service_name = trace_spans[0].resource.attributes.get(SERVICE_NAME, "unknown")

            # Session overlay: pass session_id so all turns share one file
            session_id = FileSpanExporter._get_session_id(trace_spans)
            handle, file_path, is_first_span = self._get_or_create_handle(trace_id, service_name, session_id)
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
        
        # Close handles for traces that are complete (have both root and child spans)
        traces_to_close = set()
        for trace_id in root_span_traces:
            has_child_spans = any(s.parent for s in spans_by_trace.get(trace_id, []))
            children_already_written = (
                trace_id in self.file_handles and not self._is_first_span(trace_id)
            )
            if has_child_spans or children_already_written:
                # Root + child in same batch: complete trace, close now
                traces_to_close.add(trace_id)
                self._root_span_seen.discard(trace_id)
            else:
                # Root only: child may arrive in a later batch, defer closing
                self._root_span_seen.add(trace_id)

        # Also close traces where root was seen earlier and child just arrived
        for trace_id in spans_by_trace:
            if trace_id in self._root_span_seen and trace_id not in root_span_traces:
                traces_to_close.add(trace_id)
                self._root_span_seen.discard(trace_id)

        for trace_id in traces_to_close:
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
        for session_id, (handle, file_path, _, _) in self._session_handles.items():
            try:
                if handle is not None:
                    handle.flush()
            except Exception as e:
                print(f"Error flushing session file {file_path}: {e}")
        return True

    def shutdown(self) -> None:
        """Close all file handles and stop task processor."""
        if hasattr(self, 'task_processor') and self.task_processor is not None:
            self.task_processor.stop()
        
        # Close all remaining file handles
        trace_ids_to_close = list(self.file_handles.keys())
        for trace_id in trace_ids_to_close:
            self._close_trace_handle(trace_id)

        # Close any session files that are still open
        session_ids_to_close = list(self._session_handles.keys())
        for session_id in session_ids_to_close:
            self._close_session_handle(session_id)
