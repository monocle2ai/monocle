"""
Simplified MonocleValidator for the tfwk framework.
"""
import os

from monocle_apptrace.exporters.file_exporter import (
    DEFAULT_TRACE_FOLDER,
    FileSpanExporter,
)
from monocle_apptrace.instrumentation.common.instrumentor import (
    MonocleInstrumentor,
    setup_monocle_telemetry,
)
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class MonocleValidator:
    """Simplified validator for tfwk framework."""
    
    _spans: list[Span] = []
    memory_exporter: InMemorySpanExporter = None
    file_exporter: FileSpanExporter = None
    instrumentor: MonocleInstrumentor = None
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if MonocleValidator._initialized:
            return
        test_trace_path: str = os.path.join(".", DEFAULT_TRACE_FOLDER, "test_traces")
        self.memory_exporter = InMemorySpanExporter()
        self.file_exporter = FileSpanExporter(out_path=test_trace_path)
        span_processors = [SimpleSpanProcessor(self.file_exporter), SimpleSpanProcessor(self.memory_exporter)]
        self.instrumentor = setup_monocle_telemetry(workflow_name="monocle_tfwk", span_processors=span_processors)
        MonocleValidator._initialized = True

    @property
    def spans(self):
        """Get finished spans from the memory exporter."""
        if self.memory_exporter is not None:
            return self.memory_exporter.get_finished_spans()
        return []
        
    def clear_spans(self):
        """Clear all cached spans and memory exporter spans."""
        if self.memory_exporter is not None:
            self.memory_exporter.clear()
        self._spans = []