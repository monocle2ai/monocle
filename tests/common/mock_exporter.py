from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class MockExporter(SpanExporter):
    current_trace_id: int = None
    current_file_path: str = None
    attributes_to_check: dict[str, str] = {}

    def __init__(self):
        pass

    def set_trace_check(self, attributes_to_check: dict[str, str]):
        self.attributes_to_check.update(attributes_to_check)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            for key, value in self.attributes_to_check.items():
                assert span._attributes[key] == value
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        pass
