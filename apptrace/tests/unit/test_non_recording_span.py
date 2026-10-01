
"""Monocle must still run the wrapped call when the trace is unsampled.

An unsampled parent makes every span a NonRecordingSpan (no status, attributes
or parent), which Monocle's span processing used to read and fail on.
"""
import asyncio
import os
from unittest.mock import patch

import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from opentelemetry.util._once import Once

from monocle_apptrace.instrumentation.common.instrumentor import (
    get_monocle_instrumentor,
    setup_monocle_telemetry,
)
from monocle_apptrace.instrumentation.common.wrapper import (
    atask_iter_wrapper,
    atask_wrapper,
    task_iter_wrapper,
    task_wrapper,
)
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod

TRACE_ID = 0x0AF7651916CD43DD8448EB211C80319C
PARENT_SPAN_ID = 0xB7AD6B7169203331


class Calls:
    def sync_call(self, value):
        return value * 2

    async def async_call(self, value):
        return value * 3

    def sync_stream(self, count):
        yield from range(count)

    async def async_stream(self, count):
        for i in range(count):
            yield i


def _wrapper_methods():
    methods = [
        ("sync_call", task_wrapper),
        ("async_call", atask_wrapper),
        ("sync_stream", task_iter_wrapper),
        ("async_stream", atask_iter_wrapper),
    ]
    return [WrapperMethod(package=__name__, object_name="Calls", method=method,
                          span_name=f"calls.{method}", wrapper_method=wrapper)
            for method, wrapper in methods]


def _reset_otel():
    instrumentor = get_monocle_instrumentor()
    if instrumentor is not None and instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()


@pytest.fixture
def exporter():
    _reset_otel()
    memory_exporter = InMemorySpanExporter()
    # Instrumenting imports litellm, which loads the nearest .env; keep it out of later tests.
    with patch.dict(os.environ):
        setup_monocle_telemetry(
            workflow_name="non_recording_test",
            span_processors=[SimpleSpanProcessor(memory_exporter)],
            wrapper_methods=_wrapper_methods(),
        )
        yield memory_exporter
    _reset_otel()


def _parent_flags(sampled):
    return TraceFlags(TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT)


def _run_under_parent(sampled, call):
    """Run ``call`` with a remote parent whose sampled flag is ``sampled``."""
    parent = SpanContext(trace_id=TRACE_ID, span_id=PARENT_SPAN_ID, is_remote=True,
                         trace_flags=_parent_flags(sampled))
    token = context.attach(trace.set_span_in_context(NonRecordingSpan(parent)))
    try:
        return call()
    finally:
        context.detach(token)


async def _collect(agen):
    return [item async for item in agen]


def _test_trace_spans(exporter):
    """Exported spans of the test's trace; setup can export unrelated spans."""
    return [span for span in exporter.get_finished_spans() if span.context.trace_id == TRACE_ID]


# method name -> (call, expected result)
CALLS = {
    "sync_call": (lambda: Calls().sync_call(2), 4),
    "async_call": (lambda: asyncio.run(Calls().async_call(2)), 6),
    "sync_stream": (lambda: list(Calls().sync_stream(3)), [0, 1, 2]),
    "async_stream": (lambda: asyncio.run(_collect(Calls().async_stream(3))), [0, 1, 2]),
}


@pytest.mark.parametrize("method", CALLS)
def test_unsampled_parent_runs_call_and_exports_nothing(exporter, method):
    call, expected = CALLS[method]
    assert _run_under_parent(sampled=False, call=call) == expected
    assert _test_trace_spans(exporter) == []


@pytest.mark.parametrize("method", CALLS)
def test_sampled_parent_still_traces(exporter, method):
    call, expected = CALLS[method]
    assert _run_under_parent(sampled=True, call=call) == expected
    names = [span.name for span in _test_trace_spans(exporter)]
    assert f"calls.{method}" in names, names


def _traceparent(sampled):
    return f"00-{TRACE_ID:032x}-{PARENT_SPAN_ID:016x}-{'01' if sampled else '00'}"


@pytest.fixture
def fastapi_client(exporter):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/ask")
    async def ask():
        return {"answer": "ok"}

    @app.get("/stream")
    async def stream():
        async def body():
            yield b"a"
            yield b"b"
        return StreamingResponse(body(), media_type="text/plain")

    return TestClient(app)


@pytest.mark.parametrize("path, expected", [("/ask", '{"answer":"ok"}'), ("/stream", "ab")])
def test_fastapi_unsampled_request_succeeds(fastapi_client, exporter, path, expected):
    response = fastapi_client.get(path, headers={"traceparent": _traceparent(sampled=False)})
    assert response.status_code == 200
    assert response.text == expected
    assert _test_trace_spans(exporter) == []


@pytest.mark.parametrize("path, expected", [("/ask", '{"answer":"ok"}'), ("/stream", "ab")])
def test_fastapi_sampled_request_is_traced(fastapi_client, exporter, path, expected):
    response = fastapi_client.get(path, headers={"traceparent": _traceparent(sampled=True)})
    assert response.status_code == 200
    assert response.text == expected
    spans = _test_trace_spans(exporter)
    assert any(span.attributes.get("span.type") == "http.process" for span in spans), [s.name for s in spans]
