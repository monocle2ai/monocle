import logging
import threading

from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from monocle_apptrace.exporters.base_exporter import MonocleInMemorySpanExporter
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_SCOPE_NAME

logger = logging.getLogger(__name__)

_SCOPE_ATTR = f"scope.{TRACE_RETURN_SCOPE_NAME}"


class TraceReturnSpanExporter(MonocleInMemorySpanExporter):
    """In-memory exporter that stores ONLY spans tagged with the trace-return scope."""

    def __init__(self):
        super().__init__()
        self._tr_lock = threading.Lock()

    def export(self, spans):
        tagged = [s for s in spans if s.attributes and s.attributes.get(_SCOPE_ATTR) is not None]
        if not tagged:
            return SpanExportResult.SUCCESS
        with self._tr_lock:
            return super().export(tagged)

    def pop_spans_for_trace(self, trace_id: int) -> list:
        """Return and evict all buffered spans whose trace_id matches."""
        with self._tr_lock:
            all_spans = list(self.get_finished_spans())
            matched = [s for s in all_spans if s.get_span_context().trace_id == trace_id]
            remaining = [s for s in all_spans if s.get_span_context().trace_id != trace_id]
            self.clear()
            if remaining:
                super().export(remaining)
        return matched


_trace_return_exporter = None
_singleton_lock = threading.Lock()


def get_trace_return_exporter() -> TraceReturnSpanExporter:
    global _trace_return_exporter
    with _singleton_lock:
        if _trace_return_exporter is None:
            _trace_return_exporter = TraceReturnSpanExporter()
        return _trace_return_exporter


def maybe_trace_return_processor():
    """Return a SimpleSpanProcessor around the singleton exporter, only if the feature is enabled."""
    from monocle_apptrace.instrumentation.common.trace_return import is_trace_return_enabled
    if not is_trace_return_enabled():
        return None
    return SimpleSpanProcessor(get_trace_return_exporter())
