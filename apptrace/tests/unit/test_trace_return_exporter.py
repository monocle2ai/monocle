from monocle_apptrace.exporters.trace_return_exporter import (
    TraceReturnSpanExporter,
    maybe_trace_return_processor,
)


class FakeCtx:
    def __init__(self, trace_id):
        self.trace_id = trace_id


class FakeSpan:
    def __init__(self, trace_id, tagged):
        attrs = {"monocle_apptrace.version": "1.0"}
        if tagged:
            attrs["scope.monocle_trace_return"] = "true"
        self._attributes = attrs
        self._ctx = FakeCtx(trace_id)

    @property
    def attributes(self):
        return self._attributes

    def get_span_context(self):
        return self._ctx


def test_export_keeps_only_tagged_spans():
    exp = TraceReturnSpanExporter()
    exp.export([FakeSpan(1, tagged=True), FakeSpan(1, tagged=False)])
    stored = exp.get_finished_spans()
    assert len(stored) == 1


def test_pop_spans_for_trace_evicts_by_trace_id():
    exp = TraceReturnSpanExporter()
    exp.export([FakeSpan(1, tagged=True), FakeSpan(2, tagged=True), FakeSpan(1, tagged=True)])
    popped = exp.pop_spans_for_trace(1)
    assert len(popped) == 2
    # trace 1 evicted, trace 2 remains
    remaining = exp.get_finished_spans()
    assert len(remaining) == 1
    assert remaining[0].get_span_context().trace_id == 2


def test_maybe_processor_gated_by_env(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    assert maybe_trace_return_processor() is None
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    assert maybe_trace_return_processor() is not None


def test_pop_preserves_remaining_tagged_span_without_sdk_version():
    """Regression test: remaining tagged spans without SDK version are NOT dropped on eviction."""
    exp = TraceReturnSpanExporter()

    class TaggedNoVersionSpan:
        def __init__(self, trace_id):
            self._attributes = {"scope.monocle_trace_return": "true"}  # no monocle_apptrace.version
            self._ctx = FakeCtx(trace_id)

        @property
        def attributes(self):
            return self._attributes

        def get_span_context(self):
            return self._ctx

    exp.export([TaggedNoVersionSpan(1), TaggedNoVersionSpan(2)])
    popped = exp.pop_spans_for_trace(1)
    assert len(popped) == 1
    remaining = exp.get_finished_spans()
    assert len(remaining) == 1
    assert remaining[0].get_span_context().trace_id == 2
