"""Trace context propagation for A2A calls.

The A2A SDK talks over httpx, which Monocle does not instrument, so an A2A call
carries no trace context and the agent on the other end starts its own trace.
This transport adds the context, which is what lets the test find the agent's
spans afterwards.
"""
import logging
from typing import Mapping, Optional

import httpx
from opentelemetry.context import attach, detach, get_value, set_value
from opentelemetry.propagate import inject
from opentelemetry.trace.propagation import _SPAN_KEY

from monocle_apptrace.instrumentation.common.utils import (
    _MONOCLE_SPAN_KEY,
    add_monocle_trace_state,
)

logger = logging.getLogger(__name__)


def _inject_current_span(headers: dict) -> None:
    """Write `traceparent` for the span the call is made from.

    Monocle keeps its current span under its own context key, so `inject` finds
    nothing unless that span is presented under the OpenTelemetry key first.
    `RequestSpanHandler.pre_task_processing` does the same for `requests`.
    """
    monocle_span = get_value(_MONOCLE_SPAN_KEY)
    token = attach(set_value(_SPAN_KEY, monocle_span)) if monocle_span is not None else None
    try:
        inject(headers)
    finally:
        if token is not None:
            detach(token)


def trace_headers(existing: Optional[Mapping[str, str]] = None) -> dict:
    """Return the headers an A2A call should send, lowercase-keyed.

    Adds `traceparent` and `tracestate` to the headers the request already has.
    """
    headers = {str(name).lower(): value for name, value in (existing or {}).items()}
    add_monocle_trace_state(headers)
    _inject_current_span(headers)
    return headers


class MonocleA2ATransport(httpx.AsyncBaseTransport):
    """httpx transport that adds the trace context to every request.

    Wraps another transport, so it works with a real one or with
    `httpx.MockTransport` in tests. Responses are passed through unread, so
    streaming calls keep streaming.
    """

    def __init__(self, inner: Optional[httpx.AsyncBaseTransport] = None):
        self._inner = inner if inner is not None else httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        for name, value in trace_headers(request.headers).items():
            if request.headers.get(name) != value:
                request.headers[name] = value
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def make_traced_httpx_client(*, transport: Optional[httpx.AsyncBaseTransport] = None,
                             **client_kwargs) -> httpx.AsyncClient:
    """Return an httpx client that traces the A2A calls made through it.

    Pass it to `A2AClient(httpx_client=...)`. The A2A runner uses this, and so
    can an agent under test that calls another agent over A2A.

    Args:
        transport: Transport to wrap. Defaults to a real one.
        **client_kwargs: Passed to `httpx.AsyncClient` (`timeout`, `auth`, ...).
    """
    return httpx.AsyncClient(transport=MonocleA2ATransport(transport), **client_kwargs)
