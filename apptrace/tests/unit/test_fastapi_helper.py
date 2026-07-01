import pytest

from monocle_apptrace.instrumentation.metamodel.fastapi import _helper
from monocle_apptrace.instrumentation.metamodel.fastapi.entities.http import FASTAPI_HTTP_PROCESSOR, FASTAPI_RESPONSE_PROCESSOR
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException


# StreamingState Tests
class TestStreamingState:
    def test_add_chunk_accumulates_content(self):
        state = _helper.StreamingState()
        state.add_chunk(b"hello ")
        state.add_chunk(b"world")
        
        assert state.get_captured_content() == "hello world"
        assert state.total_length == 11

    def test_add_chunk_respects_max_length(self):
        state = _helper.StreamingState(max_length=6)
        state.add_chunk(b"hello ")
        state.add_chunk(b"world!")  # This would exceed max_length
        
        # Only first chunk should be captured
        assert state.get_captured_content().endswith("...[truncated]")
        assert state.total_length == 6
        assert len(state.chunks) == 1

    def test_get_captured_content_truncates_when_exceeds_max(self):
        state = _helper.StreamingState(max_length=5)
        state.add_chunk(b"hello world")
        state.total_length = 11  # Simulate exceeding max
        
        result = state.get_captured_content()
        assert result.endswith("...[truncated]")

    def test_get_captured_content_handles_decode_error(self):
        state = _helper.StreamingState()
        state.add_chunk(b"\xff\xfe")  # Invalid UTF-8
        
        assert state.get_captured_content() == ""


# SSE Content Extraction Tests
class TestExtractSseContent:
    def test_extracts_json_events(self):
        raw = 'data: {"type": "start"}\ndata: {"type": "end"}\n'
        result = _helper._extract_sse_content(raw)
        parsed = __import__("json").loads(result)
        
        assert parsed["_sse_events"] == 2
        assert "start" in parsed["_event_types"]
        assert "end" in parsed["_event_types"]

    def test_accumulates_delta_content(self):
        raw = 'data: {"delta": "Hello "}\ndata: {"delta": "world"}\n'
        result = _helper._extract_sse_content(raw)
        parsed = __import__("json").loads(result)
        
        assert parsed["response"] == "Hello world"

    def test_handles_content_field(self):
        raw = 'data: {"content": "test"}\n'
        result = _helper._extract_sse_content(raw)
        parsed = __import__("json").loads(result)
        
        assert parsed["response"] == "test"

    def test_returns_raw_when_no_events(self):
        raw = "not sse format"
        result = _helper._extract_sse_content(raw)
        
        assert result == "not sse format"

    def test_handles_non_json_data(self):
        raw = 'data: plain text\ndata: {"type": "json"}\n'
        result = _helper._extract_sse_content(raw)
        parsed = __import__("json").loads(result)
        
        assert parsed["_sse_events"] == 2


# ASGI Middleware Tests
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


@pytest.mark.asyncio
async def test_capture_response_body_captures_status_and_body():
    scope = {}
    sent_messages = []

    async def send(message):
        sent_messages.append(message)

    wrapped_send = await _helper._capture_response_body(scope, send)
    
    await wrapped_send({"type": "http.response.start", "status": 200})
    await wrapped_send({"type": "http.response.body", "body": b'{"result": "ok"}'})

    assert scope["_response_status"] == 200
    assert scope["_response_body"] == b'{"result": "ok"}'
    assert len(sent_messages) == 2


@pytest.mark.asyncio
async def test_capture_response_body_handles_chunked_response():
    scope = {}

    async def send(message):
        pass

    wrapped_send = await _helper._capture_response_body(scope, send)
    
    await wrapped_send({"type": "http.response.start", "status": 200})
    await wrapped_send({"type": "http.response.body", "body": b'chunk1'})
    await wrapped_send({"type": "http.response.body", "body": b'chunk2'})

    assert scope["_response_body"] == b'chunk1chunk2'


# Request Extractor Tests
def test_get_url_constructs_full_url():
    scope = {
        "server": ("localhost", 8000),
        "path": "/api/test",
        "scheme": "https",
    }
    assert _helper.get_url(scope) == "https://localhost:8000/api/test"


def test_get_url_uses_defaults():
    assert _helper.get_url({}) == "http://127.0.0.1:80/"


def test_get_route_prefers_fastapi_route_template():
    route = type("Route", (), {"path": "/api/v1/items/{item_id}"})()
    scope = {"route": route, "path": "/api/v1/items/123"}

    assert _helper.get_route(scope) == "/api/v1/items/{item_id}"


def test_get_route_falls_back_to_path():
    scope = {"path": "/api/v1/items/123"}
    assert _helper.get_route(scope) == "/api/v1/items/123"


def test_get_method_extracts_method():
    assert _helper.get_method({"method": "POST"}) == "POST"
    assert _helper.get_method({}) == ""


def test_get_body_reads_buffered_request_body():
    scope = {"query_string": b"", "_request_body": b'{"question":"hello"}'}

    assert _helper.get_body(scope) == '{"question": "hello"}'


def test_get_body_returns_empty_for_missing_body():
    assert _helper.get_body({}) == ""


def test_get_body_handles_non_json():
    scope = {"_request_body": b"plain text"}
    assert _helper.get_body(scope) == "plain text"


def test_get_params_reads_query_string():
    scope = {"query_string": b"question=hello", "_request_body": b'{"question":"body"}'}

    assert _helper.get_params(scope) == "question=hello"


def test_get_params_does_not_read_buffered_request_body():
    scope = {"query_string": b"", "_request_body": b'{"question":"hello"}'}

    assert _helper.get_params(scope) == ""


def test_get_params_decodes_url_encoding():
    scope = {"query_string": b"name=hello%20world"}
    assert _helper.get_params(scope) == "name=hello world"


# Response Extractor Tests
def test_get_response_from_scope_extracts_json():
    scope = {"_response_body": b'{"result": "success"}'}
    result = _helper.get_response_from_scope(scope)
    assert result == '{"result": "success"}'


def test_get_response_from_scope_handles_sse():
    scope = {"_response_body": b'data: {"delta": "hello"}\n'}
    result = _helper.get_response_from_scope(scope)
    assert "hello" in result


def test_get_response_from_scope_returns_empty_for_missing():
    assert _helper.get_response_from_scope({}) == ""


def test_get_status_from_scope_extracts_status():
    assert _helper.get_status_from_scope({"_response_status": 200}) == "200"
    assert _helper.get_status_from_scope({"_response_status": 0}) == ""
    assert _helper.get_status_from_scope({}) == ""


def test_extract_response_from_regular_response():
    response = type("Response", (), {"body": b'{"data": "test"}'})()
    result = _helper.extract_response(response)
    assert result == '{"data": "test"}'


def test_extract_response_from_streaming_with_captured_state():
    state = _helper.StreamingState()
    state.add_chunk(b'streamed content')
    
    response = type("StreamingResponse", (), {
        "_monocle_streaming_state": state,
        "media_type": "text/plain",
    })()
    
    result = _helper.extract_response(response)
    assert result == "streamed content"


def test_extract_response_from_streaming_sse():
    state = _helper.StreamingState()
    state.add_chunk(b'data: {"delta": "hello"}\n')
    
    response = type("StreamingResponse", (), {
        "_monocle_streaming_state": state,
        "media_type": "text/event-stream",
    })()
    
    result = _helper.extract_response(response)
    assert "hello" in result


def test_extract_response_fallback_for_uncaptured_streaming():
    response = type("StreamingResponse", (), {
        "body_iterator": iter([]),
        "media_type": "text/plain",
        "status_code": 200,
    })()
    
    result = _helper.extract_response(response)
    assert '"_streaming": true' in result


# Status Extraction Tests
def test_extract_status_returns_success_code():
    response = type("Response", (), {"status_code": 200})()
    arguments = {"exception": None, "instance": response}
    
    assert _helper.extract_status(arguments) == "200"


def test_extract_status_raises_for_error_code():
    response = type("Response", (), {"status_code": 500, "body": b'{"error": "fail"}'})()
    arguments = {"exception": None, "instance": response}
    
    with pytest.raises(MonocleSpanException) as exc_info:
        _helper.extract_status(arguments)
    
    assert "500" in str(exc_info.value)


def test_extract_status_handles_streaming_error():
    response = type("StreamingResponse", (), {
        "status_code": 500,
        "body_iterator": iter([]),
    })()
    arguments = {"exception": None, "instance": response}
    
    with pytest.raises(MonocleSpanException) as exc_info:
        _helper.extract_status(arguments)
    
    assert "streaming response" in str(exc_info.value)


def test_extract_status_returns_unknown_when_no_status():
    response = type("Response", (), {})()
    arguments = {"exception": None, "instance": response}
    
    assert _helper.extract_status(arguments) == "Unknown"

# Processor Integration Tests
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
