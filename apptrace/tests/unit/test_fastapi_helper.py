import pytest

from monocle_apptrace.instrumentation.metamodel.fastapi import _helper


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


def test_get_params_reads_buffered_request_body():
    scope = {"query_string": b"", "_request_body": b'{"question":"hello"}'}

    assert _helper.get_params(scope) == '{"question":"hello"}'
