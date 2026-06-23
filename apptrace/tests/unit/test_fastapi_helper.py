import pytest

from monocle_apptrace.instrumentation.metamodel.fastapi import _helper
from monocle_apptrace.instrumentation.metamodel.fastapi.entities.http import FASTAPI_HTTP_PROCESSOR, FASTAPI_RESPONSE_PROCESSOR


@pytest.mark.asyncio
async def test_buffer_request_body_replays_messages_and_stores_body():
    messages = [
        {"type": "http.request", "body": b'{"question":', "more_body": True},
        {"type": "http.request", "body": b'"hello"}', "more_body": False},
    ]

    async def receive():
        return messages.pop(0)

    scope = {}
    replay_receive = await _helper._buffer_request_body(scope, receive)

    assert scope["_request_body"] == b'{"question":"hello"}'
    assert await replay_receive() == {
        "type": "http.request",
        "body": b'{"question":',
        "more_body": True,
    }
    assert await replay_receive() == {
        "type": "http.request",
        "body": b'"hello"}',
        "more_body": False,
    }


def test_get_route_prefers_fastapi_route_template():
    route = type("Route", (), {"path": "/api/v1/items/{item_id}"})()
    scope = {"route": route, "path": "/api/v1/items/123"}

    assert _helper.get_route(scope) == "/api/v1/items/{item_id}"


def test_get_body_reads_buffered_request_body():
    scope = {"query_string": b"", "_request_body": b'{"question":"hello"}'}

    assert _helper.get_body(scope) == '{"question":"hello"}'


def test_get_params_reads_query_string():
    scope = {"query_string": b"question=hello", "_request_body": b'{"question":"body"}'}

    assert _helper.get_params(scope) == "question=hello"


def test_get_params_does_not_read_buffered_request_body():
    scope = {"query_string": b"", "_request_body": b'{"question":"hello"}'}

    assert _helper.get_params(scope) == ""


def test_post_processors_capture_http_attributes_params_and_body():
    route = type("Route", (), {"path": "/api/v1/test"})()
    scope = {
        "method": "POST",
        "route": route,
        "path": "/api/v1/test",
        "scheme": "http",
        "server": ("0.0.0.0", 8000),
        "query_string": b"param1=value1",
        "_request_body": b'{"hello": "123"}',
    }
    arguments = {"args": (scope,)}

    http_attributes = {
        attribute["attribute"]: attribute["accessor"](arguments)
        for attribute in FASTAPI_HTTP_PROCESSOR["attributes"][0]
    }
    input_attributes = {
        attribute["attribute"]: attribute["accessor"](arguments)
        for event in FASTAPI_RESPONSE_PROCESSOR["events"]
        if event["name"] == "data.input"
        for attribute in event["attributes"]
    }

    assert http_attributes == {
        "method": "POST",
        "route": "/api/v1/test",
        "url": "http://0.0.0.0:8000/api/v1/test",
    }
    assert input_attributes == {
        "params": "param1=value1",
        "request_body": '{"hello": "123"}',
    }
