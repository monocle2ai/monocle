from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common import instrumentor as inst


def test_processor_added_when_enabled(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    # helper under test: appends the trace-return processor when enabled
    result = inst._append_trace_return_processor([])
    assert len(result) == 1
    assert isinstance(result[0], SimpleSpanProcessor)


def test_processor_absent_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    result = inst._append_trace_return_processor([])
    assert result == []
