"""Appending an invocation's spans to the AgentCore response.

AgentCore serializes whatever the entrypoint returned, so there is no send-time
hook like FastAPI's. The trailer goes onto the Response object built by
_handle_invocation instead, which means Content-Length has to be corrected or
the trailer is truncated off the wire.
"""
import pytest
from starlette.responses import Response, StreamingResponse

from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER
from monocle_apptrace.instrumentation.metamodel.agentcore.agentcore_handler import (
    AgentCoreSpanHandler,
)

AGENT_BODY = b'"Flight booked."'
DELIMITER = "__MONOCLE_TRACES__deadbeef__"
PAYLOAD = "H4sIAAAAAAAAA6tWSkxKTlEqSk1MUbJSykjNyclXqgUAr6nQVRUAAAA="


class FakeSpanContext:
    trace_id = 12345


class FakeSpan:
    def get_span_context(self):
        return FakeSpanContext()


@pytest.fixture
def trailer_available(monkeypatch):
    """Pretend this trace captured spans, so a trailer is produced."""
    monkeypatch.setattr(tr, "is_trace_return_enabled", lambda: True)
    monkeypatch.setattr(
        tr, "get_response_trailer",
        lambda trace_id: (tr.build_response_header_value(DELIMITER),
                          (DELIMITER + PAYLOAD).encode("utf-8")),
    )


@pytest.fixture
def no_trailer(monkeypatch):
    """Trace-return on, but this trace produced nothing to return."""
    monkeypatch.setattr(tr, "is_trace_return_enabled", lambda: True)
    monkeypatch.setattr(tr, "get_response_trailer", lambda trace_id: None)


def test_trailer_is_appended_to_the_response(trailer_available):
    response = Response(AGENT_BODY, media_type="application/json")

    AgentCoreSpanHandler._inject_trailer(response, FakeSpan())

    assert response.body.startswith(AGENT_BODY), "the agent's own body must be preserved"
    assert DELIMITER.encode() in response.body
    assert PAYLOAD.encode() in response.body


def test_content_length_is_corrected(trailer_available):
    """Starlette fixes Content-Length at construction; a stale value truncates."""
    response = Response(AGENT_BODY, media_type="application/json")

    AgentCoreSpanHandler._inject_trailer(response, FakeSpan())

    assert response.headers["content-length"] == str(len(response.body))
    assert int(response.headers["content-length"]) > len(AGENT_BODY)


def test_response_header_advertises_the_delimiter(trailer_available):
    response = Response(AGENT_BODY, media_type="application/json")

    AgentCoreSpanHandler._inject_trailer(response, FakeSpan())

    assert tr.parse_delimiter_from_header(
        response.headers[TRACE_RETURN_RESPONSE_HEADER]) == DELIMITER


def test_untouched_when_trace_return_is_disabled(monkeypatch):
    """Default deployments must behave exactly as before."""
    monkeypatch.setattr(tr, "is_trace_return_enabled", lambda: False)
    response = Response(AGENT_BODY, media_type="application/json")

    AgentCoreSpanHandler._inject_trailer(response, FakeSpan())

    assert response.body == AGENT_BODY
    assert TRACE_RETURN_RESPONSE_HEADER not in response.headers


def test_untouched_when_this_trace_captured_no_spans(no_trailer):
    response = Response(AGENT_BODY, media_type="application/json")

    AgentCoreSpanHandler._inject_trailer(response, FakeSpan())

    assert response.body == AGENT_BODY
    assert TRACE_RETURN_RESPONSE_HEADER not in response.headers


def test_streaming_responses_are_skipped(trailer_available):
    """A streaming response has no body to extend."""
    async def gen():
        yield b"chunk"

    response = StreamingResponse(gen(), media_type="text/event-stream")

    AgentCoreSpanHandler._inject_trailer(response, FakeSpan())

    assert TRACE_RETURN_RESPONSE_HEADER not in response.headers


def test_injection_failure_never_breaks_the_response(monkeypatch):
    """Returning traces is diagnostic; a failure must not affect the answer."""
    monkeypatch.setattr(tr, "is_trace_return_enabled", lambda: True)
    monkeypatch.setattr(tr, "get_response_trailer",
                        lambda trace_id: (_ for _ in ()).throw(RuntimeError("boom")))
    response = Response(AGENT_BODY, media_type="application/json")

    handler = AgentCoreSpanHandler()
    handler.post_task_processing(
        {}, None, None, (), {}, response, None, FakeSpan(), None)

    assert response.body == AGENT_BODY


if __name__ == "__main__":
    pytest.main([__file__])
