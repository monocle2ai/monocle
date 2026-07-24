from monocle_apptrace.instrumentation.common.span_handler import HttpSpanHandler
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.exporters.trace_return_exporter import get_trace_return_exporter


class FakeCtx:
    def __init__(self, trace_id): self.trace_id = trace_id


class FakeSpan:
    def __init__(self, trace_id):
        self._attributes = {"monocle_apptrace.version": "1.0", "scope.monocle_trace_return": "true"}
        self._ctx = FakeCtx(trace_id)
    @property
    def attributes(self): return self._attributes
    def get_span_context(self): return self._ctx
    def to_json(self): return '{"name": "inference"}'


def test_build_trailer_returns_bytes_for_matching_trace():
    exp = get_trace_return_exporter()
    exp.clear()
    exp.export([FakeSpan(42)])
    delim = tr.make_delimiter()
    trailer = HttpSpanHandler().build_trace_return_trailer(42, delim)
    assert trailer is not None
    clean, payload = tr.split_body_and_trailer(b"BODY" + trailer, delim)
    assert clean == b"BODY"
    import json
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "inference"
    # eviction: a second call for the same trace now finds nothing
    assert HttpSpanHandler().build_trace_return_trailer(42, tr.make_delimiter()) is None


def test_build_trailer_none_when_no_spans():
    exp = get_trace_return_exporter()
    exp.clear()
    assert HttpSpanHandler().build_trace_return_trailer(99, tr.make_delimiter()) is None
