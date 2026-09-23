"""Unit tests for the httpx transport behind the A2A runner.

httpx's MockTransport stands in for the agent, so these need no server.
"""
import httpx
import pytest

from monocle_test_tools.a2a_transport import (
    MonocleA2ATransport,
    make_traced_httpx_client,
    trace_headers,
)


# -- trace_headers ---------------------------------------------------------

def test_trace_headers_carry_trace_context():
    headers = trace_headers({})
    assert "tracestate" in headers


def test_traceparent_names_the_monocle_span_the_call_is_made_from():
    """Without this the agent starts its own trace and neither lookup finds it.

    Monocle keeps its span under its own context key, so a plain `inject`
    writes no traceparent at all.
    """
    from opentelemetry.context import attach, detach, set_value
    from opentelemetry.sdk.trace import TracerProvider

    from monocle_apptrace.instrumentation.common.utils import _MONOCLE_SPAN_KEY

    tracer = TracerProvider().get_tracer(__name__)
    with tracer.start_as_current_span("a2a.client.send_message") as span:
        token = attach(set_value(_MONOCLE_SPAN_KEY, span))
        try:
            headers = trace_headers({})
        finally:
            detach(token)

    context = span.get_span_context()
    assert headers["traceparent"].split("-")[1] == format(context.trace_id, "032x")
    assert headers["traceparent"].split("-")[2] == format(context.span_id, "016x")


def test_trace_headers_without_a_monocle_span_carry_no_parent():
    assert "traceparent" not in trace_headers({})


def test_trace_headers_keep_what_the_caller_set():
    headers = trace_headers({"X-Caller": "mine"})
    assert headers["x-caller"] == "mine"


# -- the transport ---------------------------------------------------------

@pytest.mark.asyncio
async def test_transport_sends_trace_context():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json={})

    async with make_traced_httpx_client(transport=httpx.MockTransport(handler)) as client:
        await client.post("http://agent.test/", json={})

    assert "tracestate" in seen


@pytest.mark.asyncio
async def test_transport_passes_the_response_through_untouched():
    mock = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"jsonrpc": "2.0"}))
    async with make_traced_httpx_client(transport=mock) as client:
        response = await client.post("http://agent.test/", json={})

    assert response.json() == {"jsonrpc": "2.0"}


@pytest.mark.asyncio
async def test_transport_defaults_to_wrapping_a_real_transport():
    transport = MonocleA2ATransport()
    assert isinstance(transport._inner, httpx.AsyncHTTPTransport)
    await transport.aclose()
