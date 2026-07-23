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


def test_pop_and_build_trailer_roundtrip():
    exp = get_trace_return_exporter(); exp.clear()
    exp.export([FakeSpan(7)])
    delim = tr.make_delimiter()
    trailer = tr.pop_and_build_trailer(7, delim)
    assert trailer is not None
    clean, payload = tr.split_body_and_trailer(b"BODY" + trailer, delim)
    assert clean == b"BODY"
    import json
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "inference"
    # evicted
    assert tr.pop_and_build_trailer(7, tr.make_delimiter()) is None


def test_get_response_trailer_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    get_trace_return_exporter().clear()
    assert tr.get_response_trailer(7) is None


def test_get_response_trailer_enabled(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear()
    exp.export([FakeSpan(9)])
    result = tr.get_response_trailer(9)
    assert result is not None
    header_value, trailer = result
    assert header_value.startswith("v1;")
    delim = tr.parse_delimiter_from_header(header_value)
    clean, payload = tr.split_body_and_trailer(b"X" + trailer, delim)
    assert clean == b"X"


def test_get_response_trailer_enabled_no_spans(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    get_trace_return_exporter().clear()
    assert tr.get_response_trailer(123456) is None
