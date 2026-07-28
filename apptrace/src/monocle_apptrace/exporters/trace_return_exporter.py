import threading

from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from monocle_apptrace.exporters.base_exporter import MonocleInMemorySpanExporter
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_SCOPE_NAME

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
            return InMemorySpanExporter.export(self, tagged)

    def shutdown(self) -> None:
        # This is a process-global singleton registered on the tracer provider via a
        # SpanProcessor (see maybe_trace_return_processor). Provider teardown --
        # reset_span_processors / clear_span_processors, or a subsequent
        # setup_monocle_telemetry -- propagates shutdown() to it. The base
        # InMemorySpanExporter.shutdown() would set _stopped=True, after which export()
        # silently returns FAILURE and drops every span, permanently disabling
        # trace-return for the rest of the process. Keep the singleton usable across
        # provider teardowns instead of stopping it.
        with self._tr_lock:
            self._stopped = False

    def pop_spans_for_trace(self, trace_id: int) -> list:
        """Return and evict all buffered spans whose trace_id matches."""
        with self._tr_lock:
            all_spans = list(self.get_finished_spans())
            matched = [s for s in all_spans if s.get_span_context().trace_id == trace_id]
            remaining = [s for s in all_spans if s.get_span_context().trace_id != trace_id]
            self.clear()
            if remaining:
                InMemorySpanExporter.export(self, remaining)
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
