from opentelemetry.sdk.trace.export import ConsoleSpanExporter


class CustomConsoleSpanExporter(ConsoleSpanExporter):
    def __init__(self):
        super().__init__()
        self.captured_spans = []

    def export(self, spans):
        self.captured_spans.extend(spans)
        super().export(spans)

    def get_captured_spans(self):
        return self.captured_spans
