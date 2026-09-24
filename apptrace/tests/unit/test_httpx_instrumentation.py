"""Unit tests for the httpx client instrumentation.

The handler is exercised directly with real httpx objects, so these need no
server and no instrumented process.
"""
import base64
import gzip
import json

import pytest

httpx = pytest.importorskip("httpx")

from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER
from monocle_apptrace.instrumentation.metamodel.httpx import _helper

SPAN_JSON = json.dumps([{"name": "book_flight"}])


def _handler():
    return _helper.HttpxSpanHandler()


def _arguments(request=None, result=None):
    return {"args": (request,) if request is not None else (), "kwargs": {},
            "result": result}


def _response_with_spans(body: bytes) -> httpx.Response:
    """A response shaped like one from a server with trace return enabled."""
    delimiter = tr.make_delimiter()
    trailer = base64.b64encode(gzip.compress(SPAN_JSON.encode("utf-8")))
    return httpx.Response(
        200,
        headers={TRACE_RETURN_RESPONSE_HEADER: tr.build_response_header_value(delimiter)},
        content=body + delimiter.encode("utf-8") + trailer,
    )


# -- what the span records -------------------------------------------------

def test_the_call_is_described_from_the_request():
    request = httpx.Request("POST", "http://agent.test/rpc?turn=2")
    arguments = _arguments(request)

    assert _helper.get_method(arguments) == "POST"
    assert _helper.get_route(arguments) == "agent.test/rpc"
    assert _helper.get_params(arguments) == "turn=2"


def test_the_request_is_found_when_passed_by_name():
    request = httpx.Request("GET", "http://agent.test/")
    assert _helper.get_request({"args": (), "kwargs": {"request": request}}) is request


def test_the_response_is_described_by_status_and_body():
    response = httpx.Response(200, json={"ok": True})
    assert _helper.extract_status(response) == "200"
    assert "ok" in _helper.extract_response(response)


# -- which calls are traced ------------------------------------------------

def test_calls_are_skipped_unless_their_host_was_asked_for(monkeypatch):
    monkeypatch.setattr(_helper, "allowed_urls", [])
    request = httpx.Request("GET", "http://agent.test/")

    assert _helper.httpx_skip_span(request, trace_all_urls=False) is True
    assert _helper.httpx_skip_span(request, trace_all_urls=True) is False


def test_an_allowed_host_is_traced(monkeypatch):
    monkeypatch.setattr(_helper, "allowed_urls", ["http://agent.test"])

    assert _helper.httpx_skip_span(httpx.Request("GET", "http://agent.test/x"),
                                   trace_all_urls=False) is False
    assert _helper.httpx_skip_span(httpx.Request("GET", "http://other.test/x"),
                                   trace_all_urls=False) is True


# -- trace context out -----------------------------------------------------

def test_the_trace_context_is_put_on_the_request():
    request = httpx.Request("POST", "http://agent.test/")
    _handler().pre_task_processing(None, None, None, (request,), {}, None)

    assert "tracestate" in request.headers


def test_the_caller_keeps_its_own_headers():
    request = httpx.Request("POST", "http://agent.test/", headers={"x-caller": "mine"})
    _handler().pre_task_processing(None, None, None, (request,), {}, None)

    assert request.headers["x-caller"] == "mine"


# -- returned spans back ---------------------------------------------------

def test_returned_spans_come_off_the_response():
    body = b'{"jsonrpc":"2.0"}'
    response = _response_with_spans(body)

    _handler().post_task_processing(None, None, None, (), {}, response, None, None, None)

    assert response.content == body                     # the answer is intact
    assert json.loads(response._monocle_remote_spans)[0]["name"] == "book_flight"


def test_a_plain_response_is_left_alone():
    response = httpx.Response(200, json={"jsonrpc": "2.0"})

    _handler().post_task_processing(None, None, None, (), {}, response, None, None, None)

    assert response.json() == {"jsonrpc": "2.0"}
    assert not hasattr(response, "_monocle_remote_spans")


def test_returned_spans_reach_a_collector():
    """A caller that never sees the response can still pick the spans up."""
    response = _response_with_spans(b"{}")

    with tr.collect_returned_spans() as returned:
        _handler().post_task_processing(None, None, None, (), {}, response, None, None, None)

    assert len(returned) == 1
    assert json.loads(returned[0])[0]["name"] == "book_flight"


def test_nothing_is_collected_outside_a_block():
    response = _response_with_spans(b"{}")

    _handler().post_task_processing(None, None, None, (), {}, response, None, None, None)

    with tr.collect_returned_spans() as returned:
        pass
    assert returned == []
