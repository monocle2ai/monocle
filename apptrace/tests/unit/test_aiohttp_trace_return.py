import asyncio
import json
import pytest
aiohttp_web = pytest.importorskip("aiohttp.web")
from monocle_apptrace.instrumentation.metamodel.aiohttp import _helper as ah
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
    def to_json(self): return '{"name": "aiohttp_child"}'


def _run(coro): return asyncio.new_event_loop().run_until_complete(coro)


def test_aiohttp_buffered_injection(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(41)])
    resp = aiohttp_web.Response(body=json.dumps({"answer": "hi"}).encode("utf-8"),
                                content_type="application/json")
    ah._aiohttp_inject_buffered(resp, trace_id=41)
    hv = resp.headers.get("x-monocle-traces")
    assert hv is not None
    delim = tr.parse_delimiter_from_header(hv)
    clean, payload = tr.split_body_and_trailer(bytes(resp.body), delim)
    assert json.loads(clean.decode())["answer"] == "hi"
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "aiohttp_child"


def test_aiohttp_buffered_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    get_trace_return_exporter().clear()
    resp = aiohttp_web.Response(body=b"{}", content_type="application/json")
    ah._aiohttp_inject_buffered(resp, trace_id=41)
    assert bytes(resp.body) == b"{}"
    assert resp.headers.get("x-monocle-traces") is None


class FakeStreamResponse:
    """Stand-in for aiohttp.web.StreamResponse: exercises our wrapper logic
    (header on prepare, trailer write on write_eof) without needing a real
    aiohttp request/transport."""
    def __init__(self):
        self.prepared = False
        self.headers = {}
        self.writes = []

    async def write(self, data):
        self.writes.append(data)


def test_aiohttp_stream_prepare_sets_header_when_eligible(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    token = set_scope(TRACE_RETURN_SCOPE_NAME, "true")
    try:
        instance = FakeStreamResponse()

        async def wrapped(*args, **kwargs):
            instance.prepared = True
            return "prepare-result"

        inner = ah.aiohttp_streamresponse_prepare(None, None, {})
        result = _run(inner(wrapped, instance, ("fake_request",), {}))

        assert result == "prepare-result"
        assert "x-monocle-traces" in instance.headers
        delim = tr.parse_delimiter_from_header(instance.headers["x-monocle-traces"])
        assert delim == instance._monocle_tr_delimiter
    finally:
        remove_scope(token)


def test_aiohttp_stream_prepare_noop_when_already_prepared(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    token = set_scope(TRACE_RETURN_SCOPE_NAME, "true")
    try:
        instance = FakeStreamResponse()
        instance.prepared = True  # simulate prepare() already called once

        async def wrapped(*args, **kwargs):
            return "prepare-result"

        inner = ah.aiohttp_streamresponse_prepare(None, None, {})
        _run(inner(wrapped, instance, ("fake_request",), {}))

        assert "x-monocle-traces" not in instance.headers
        assert not hasattr(instance, "_monocle_tr_delimiter")
    finally:
        remove_scope(token)


def test_aiohttp_stream_prepare_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    instance = FakeStreamResponse()

    async def wrapped(*args, **kwargs):
        instance.prepared = True
        return "prepare-result"

    inner = ah.aiohttp_streamresponse_prepare(None, None, {})
    _run(inner(wrapped, instance, ("fake_request",), {}))

    assert "x-monocle-traces" not in instance.headers


def test_aiohttp_stream_write_eof_writes_trailer_before_eof(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(55)])
    monkeypatch.setattr(ah, "get_current_monocle_span", lambda: FakeSpan(55))

    instance = FakeStreamResponse()
    delim = tr.make_delimiter()
    instance._monocle_tr_delimiter = delim

    eof_calls = []

    async def wrapped(*args, **kwargs):
        eof_calls.append("eof")
        return "eof-result"

    inner = ah.aiohttp_streamresponse_write_eof(None, None, {})
    result = _run(inner(wrapped, instance, (), {}))

    assert result == "eof-result"
    assert instance.writes, "expected the trailer to be written"
    assert eof_calls == ["eof"], "wrapped write_eof should still be invoked"
    body = b"".join(instance.writes)
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert clean == b""
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "aiohttp_child"
    # delimiter cleared after use, guarding against double-injection
    assert instance._monocle_tr_delimiter is None


def test_aiohttp_stream_write_eof_noop_when_no_delimiter(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    instance = FakeStreamResponse()  # no _monocle_tr_delimiter set (prepare never ran)

    async def wrapped(*args, **kwargs):
        return "eof-result"

    inner = ah.aiohttp_streamresponse_write_eof(None, None, {})
    result = _run(inner(wrapped, instance, (), {}))

    assert result == "eof-result"
    assert instance.writes == []
