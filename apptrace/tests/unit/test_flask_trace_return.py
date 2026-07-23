import json
import pytest
werkzeug = pytest.importorskip("werkzeug")
from werkzeug.wrappers import Response
from werkzeug.test import EnvironBuilder
from monocle_apptrace.instrumentation.metamodel.flask import _helper as fh
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_SCOPE_NAME
from monocle_apptrace.instrumentation.common.utils import set_scope, remove_scope
from monocle_apptrace.exporters.trace_return_exporter import get_trace_return_exporter


class FakeCtx:
    def __init__(self, tid): self.trace_id = tid
class FakeSpan:
    def __init__(self, tid):
        self._a = {"monocle_apptrace.version": "1.0", "scope.monocle_trace_return": "true"}
        self._c = FakeCtx(tid)
    @property
    def attributes(self): return self._a
    def get_span_context(self): return self._c
    def to_json(self): return '{"name": "flask_child"}'


def _collect_start_response():
    captured = {}
    def start_response(status, headers, exc_info=None):
        captured["status"] = status
        captured["headers"] = headers
    return captured, start_response


def test_flask_wrap_start_response_adds_header_and_bumps_content_length():
    captured, start_response = _collect_start_response()
    header_value = "v1; delim=__MONOCLE_TRACES__abc__"
    wrapped_sr = fh._flask_wrap_start_response(start_response, header_value, extra_len=10)
    wrapped_sr("200 OK", [("Content-Length", "16"), ("Content-Type", "application/json")])
    headers = dict(captured["headers"])
    assert headers["Content-Length"] == "26"
    assert headers["x-monocle-traces"] == header_value
    assert captured["status"] == "200 OK"


def test_flask_wrap_start_response_noop_content_length_when_extra_len_zero():
    captured, start_response = _collect_start_response()
    header_value = "v1; delim=__MONOCLE_TRACES__xyz__"
    wrapped_sr = fh._flask_wrap_start_response(start_response, header_value, extra_len=0)
    wrapped_sr("200 OK", [("Content-Type", "text/event-stream")])
    headers = dict(captured["headers"])
    assert "Content-Length" not in headers
    assert headers["x-monocle-traces"] == header_value


def test_flask_chain_trailer_appends_after_body_chunks():
    real_iter = iter([b"chunk-a", b"chunk-b"])
    trailer = b"__MONOCLE_TRACES__abc__PAYLOAD"
    wrapped = list(fh._flask_chain_trailer(real_iter, trailer))
    assert wrapped == [b"chunk-a", b"chunk-b", trailer]


def test_flask_chain_trailer_no_trailer_passes_through():
    real_iter = iter([b"chunk-a", b"chunk-b"])
    wrapped = list(fh._flask_chain_trailer(real_iter, None))
    assert wrapped == [b"chunk-a", b"chunk-b"]


def test_flask_streaming_injection(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(32)])
    delim = tr.make_delimiter()
    real_iter = iter([b"chunk-a", b"chunk-b"])
    wrapped = list(fh._flask_wrap_stream(real_iter, trace_id=32, delimiter=delim))
    body = b"".join(wrapped)
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert clean == b"chunk-achunk-b"
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "flask_child"


def test_flask_buffered_wire_injection_keeps_server_span_clean(monkeypatch):
    """Regression guard: the wire-level (start_response/app_iter) injection must
    never mutate the werkzeug Response object. FlaskResponseSpanHandler reads
    instance.get_data() to populate the response span's data.output event, so if
    the trailer were appended to the Response body (as an earlier, rejected
    approach did), the span's recorded output would be corrupted with the raw
    trailer bytes. This test drives the real werkzeug Response.__call__ (the
    'wrapped' the wrapper delegates to) through our wire-level helpers and
    asserts the Response body stays byte-identical to the original clean JSON."""
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(33)])
    resp = Response(json.dumps({"answer": "hi"}), mimetype="application/json")
    clean_body = resp.get_data()

    header_value, trailer = tr.get_response_trailer(trace_id=33)

    environ = {"REQUEST_METHOD": "GET", "SERVER_NAME": "test", "SERVER_PORT": "80",
               "wsgi.url_scheme": "http", "wsgi.input": None, "wsgi.errors": None,
               "PATH_INFO": "/"}
    captured, start_response = _collect_start_response()
    wrapped_sr = fh._flask_wrap_start_response(start_response, header_value, len(trailer))
    app_iter = resp(environ, wrapped_sr)
    wire_body = b"".join(fh._flask_chain_trailer(app_iter, trailer))

    # The wire-level output the client actually receives carries the trailer...
    wire_clean, wire_payload = tr.split_body_and_trailer(wire_body, tr.parse_delimiter_from_header(header_value))
    assert wire_clean == clean_body
    assert json.loads(tr.decode_payload(wire_payload))[0]["name"] == "flask_child"
    # ...but the Response object the span handler reads was never touched.
    assert resp.get_data() == clean_body
    assert b"__MONOCLE_TRACES__" not in resp.get_data()
    assert fh.extract_response(resp) == json.dumps({"answer": "hi"})


def test_flask_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    get_trace_return_exporter().clear()
    assert tr.get_response_trailer(trace_id=31) is None


def _stub_monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs):
    """Test double for monocle_wrapper: bypasses real OTel span creation entirely and
    just invokes the real wrapped call (e.g. werkzeug's actual Response.__call__), so
    these tests exercise flask_response_wrapper's own eligibility gate, branch
    selection, and args substitution (new start_response fed through) without needing
    a real tracer."""
    return wrapped(*args, **kwargs)


def test_flask_response_wrapper_eligible_buffered(monkeypatch):
    """End-to-end coverage of the composed flask_response_wrapper for the buffered
    case: is_trace_return_enabled() True + is_scope_set(TRACE_RETURN_SCOPE_NAME) True
    exercises the eligible+buffered branch, which substitutes a wrapped start_response
    into the args handed to monocle_wrapper. Asserts the recorded start_response saw
    the header + bumped Content-Length, the returned iterable ends with the trailer,
    and (regression guard) the Response object itself stays untouched."""
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    monkeypatch.setattr(fh, "monocle_wrapper", _stub_monocle_wrapper)
    monkeypatch.setattr(fh, "get_current_monocle_span", lambda: FakeSpan(41))
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(41)])
    token = set_scope(TRACE_RETURN_SCOPE_NAME, "true")
    try:
        resp = Response(json.dumps({"answer": "hi"}), mimetype="application/json")
        clean_body = resp.get_data()
        environ = EnvironBuilder(method="GET", path="/").get_environ()
        captured, start_response = _collect_start_response()

        inner = fh.flask_response_wrapper(None, None, {})
        app_iter = inner(resp.__call__, resp, (environ, start_response), {})
        body = b"".join(app_iter)

        headers = dict(captured["headers"])
        assert "x-monocle-traces" in headers
        assert int(headers["Content-Length"]) == len(body)
        delim = tr.parse_delimiter_from_header(headers["x-monocle-traces"])
        clean, payload = tr.split_body_and_trailer(body, delim)
        assert clean == clean_body
        assert json.loads(tr.decode_payload(payload))[0]["name"] == "flask_child"
        # regression guard: the Response object the span handler reads stays clean
        assert resp.get_data() == clean_body
    finally:
        remove_scope(token)


def test_flask_response_wrapper_eligible_streamed(monkeypatch):
    """End-to-end coverage of the eligible+streamed branch: header is set up front
    with no Content-Length adjustment (streamed responses don't carry one), and the
    trailer is appended as the final chunk of the returned iterable."""
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    monkeypatch.setattr(fh, "monocle_wrapper", _stub_monocle_wrapper)
    monkeypatch.setattr(fh, "get_current_monocle_span", lambda: FakeSpan(42))
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(42)])
    token = set_scope(TRACE_RETURN_SCOPE_NAME, "true")
    try:
        def gen():
            yield b"chunk-a"
            yield b"chunk-b"
        resp = Response(gen())
        assert resp.is_streamed
        environ = EnvironBuilder(method="GET", path="/").get_environ()
        captured, start_response = _collect_start_response()

        inner = fh.flask_response_wrapper(None, None, {})
        app_iter = inner(resp.__call__, resp, (environ, start_response), {})
        body = b"".join(app_iter)

        headers = dict(captured["headers"])
        assert "x-monocle-traces" in headers
        assert "Content-Length" not in headers
        delim = tr.parse_delimiter_from_header(headers["x-monocle-traces"])
        clean, payload = tr.split_body_and_trailer(body, delim)
        assert clean == b"chunk-achunk-b"
        assert json.loads(tr.decode_payload(payload))[0]["name"] == "flask_child"
    finally:
        remove_scope(token)


def test_flask_response_wrapper_not_eligible_passthrough(monkeypatch):
    """When the scope isn't set (even with the feature enabled), flask_response_wrapper
    must be a pure no-op: delegate straight to monocle_wrapper with the original,
    un-substituted args -- no header, no Content-Length change, body byte-identical."""
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    monkeypatch.setattr(fh, "monocle_wrapper", _stub_monocle_wrapper)
    # scope intentionally left unset -> is_scope_set(TRACE_RETURN_SCOPE_NAME) is False
    assert not fh.is_scope_set(TRACE_RETURN_SCOPE_NAME)

    resp = Response(json.dumps({"answer": "hi"}), mimetype="application/json")
    clean_body = resp.get_data()
    environ = EnvironBuilder(method="GET", path="/").get_environ()
    captured, start_response = _collect_start_response()

    inner = fh.flask_response_wrapper(None, None, {})
    app_iter = inner(resp.__call__, resp, (environ, start_response), {})
    body = b"".join(app_iter)

    headers = dict(captured["headers"])
    assert "x-monocle-traces" not in headers
    assert int(headers["Content-Length"]) == len(clean_body)
    assert body == clean_body
    assert resp.get_data() == clean_body
