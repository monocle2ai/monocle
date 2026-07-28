import asyncio
import pytest
from monocle_apptrace.instrumentation.metamodel.fastapi import _helper as h
from monocle_apptrace.instrumentation.common import trace_return as tr


class StubHandler:
    def __init__(self, trailer): self._trailer = trailer
    def build_trace_return_trailer(self, trace_id, delimiter): return self._trailer


def _run(coro): return asyncio.new_event_loop().run_until_complete(coro)


def _collect_send():
    sent = []
    async def send(message): sent.append(message)
    return sent, send


def test_buffered_injection_appends_trailer_and_fixes_length(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    delim = "\n__MONOCLE_TRACES__x__"
    trailer = delim.encode() + b"PAYLOAD"
    sent, send = _collect_send()
    scope = {"headers": [(b"x-monocle-retrieve-traces", b"true")]}
    wrapped = _run(h._inject_trace_return_send(scope, send, StubHandler(trailer), trace_id=7, delimiter=delim))
    _run(wrapped({"type": "http.response.start", "status": 200,
                  "headers": [(b"content-length", b"16"), (b"content-type", b"application/json")]}))
    _run(wrapped({"type": "http.response.body", "body": b'{"answer": "hi"}', "more_body": False}))
    start = [m for m in sent if m["type"] == "http.response.start"][0]
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    headers = dict(start["headers"])
    assert headers[b"content-length"] == str(16 + len(trailer)).encode()
    assert any(k == b"x-monocle-traces" for k, _ in start["headers"])
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert clean == b'{"answer": "hi"}'
    assert payload == "PAYLOAD"


def test_streaming_injection_appends_trailer_chunk(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    delim = "\n__MONOCLE_TRACES__y__"
    trailer = delim.encode() + b"PAYLOAD"
    sent, send = _collect_send()
    scope = {"headers": [(b"x-monocle-retrieve-traces", b"true")]}
    wrapped = _run(h._inject_trace_return_send(scope, send, StubHandler(trailer), trace_id=7, delimiter=delim))
    _run(wrapped({"type": "http.response.start", "status": 200,
                  "headers": [(b"content-type", b"text/event-stream")]}))
    _run(wrapped({"type": "http.response.body", "body": b"data: a\n\n", "more_body": True}))
    _run(wrapped({"type": "http.response.body", "body": b"data: b\n\n", "more_body": False}))
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert clean == b"data: a\n\ndata: b\n\n"
    assert payload == "PAYLOAD"
    # start forwarded with header, and stream not buffered (start before bodies)
    assert sent[0]["type"] == "http.response.start"


def test_no_injection_when_no_trailer(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    delim = "\n__MONOCLE_TRACES__z__"
    sent, send = _collect_send()
    scope = {"headers": [(b"x-monocle-retrieve-traces", b"true")]}
    wrapped = _run(h._inject_trace_return_send(scope, send, StubHandler(None), trace_id=7, delimiter=delim))
    _run(wrapped({"type": "http.response.start", "status": 200,
                  "headers": [(b"content-length", b"3")]}))
    _run(wrapped({"type": "http.response.body", "body": b"abc", "more_body": False}))
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert body == b"abc"


def test_streaming_no_trailer_passes_chunks_through(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    delim = "\n__MONOCLE_TRACES__w__"
    sent, send = _collect_send()
    scope = {"headers": [(b"x-monocle-retrieve-traces", b"true")]}
    wrapped = _run(h._inject_trace_return_send(scope, send, StubHandler(None), trace_id=7, delimiter=delim))
    _run(wrapped({"type": "http.response.start", "status": 200,
                  "headers": [(b"content-type", b"text/event-stream")]}))
    _run(wrapped({"type": "http.response.body", "body": b"data: a\n\n", "more_body": True}))
    _run(wrapped({"type": "http.response.body", "body": b"data: b\n\n", "more_body": False}))
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert body == b"data: a\n\ndata: b\n\n"
    assert delim.encode() not in body
    assert sent[0]["type"] == "http.response.start"
