from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SpanExportResult


class CustomConsoleSpanExporter(ConsoleSpanExporter):
    def __init__(self):
        super().__init__()
        self.captured_spans = []

    def export(self, spans):
        self.captured_spans.extend(spans)
        try:
            super().export(spans)
        except ValueError as e:
            if "I/O operation on closed file" in str(e):
                # File was closed, but we still captured the spans
                pass
            else:
                raise
        return SpanExportResult.SUCCESS

    def get_captured_spans(self):
        return self.captured_spans
    
    def reset(self):
        self.captured_spans.clear()
