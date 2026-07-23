import os

# MONOCLE_ENABLE_TRACE_RETURN must be "true" *before* MonocleValidator() is
# constructed anywhere in this process: setup_monocle_telemetry() only wires
# up the TraceReturnSpanExporter processor once, at instrumentor-setup time,
# and MonocleValidator is a session-wide singleton. Setting it here at module
# import time guarantees it is in place before the first construction, as
# long as this file is run on its own / is the first test module to touch
# MonocleValidator in the pytest session.
os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "true"
# Server-side key (default_trace_retrieval_callback checks the incoming
# x-monocle-retrieve-traces header against this) and client-side key
# (HttpRunner._maybe_inject_retrieval_key reads this to auto-inject the
# header). Same value == authorized; must be set before MonocleValidator()
# is constructed for the same reason as MONOCLE_ENABLE_TRACE_RETURN above.
os.environ["MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY"] = "e2e-s3cret"
os.environ["MONOCLE_TRACE_RETRIEVAL_KEY"] = "e2e-s3cret"

import socket
import threading
import time

import pytest

pytest_plugins = ["monocle_test_tools.pytest_plugin"]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server():
    import asyncio

    import uvicorn
    from fastapi import FastAPI
    from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method

    app = FastAPI()

    @monocle_trace_method(span_name="answer_question")
    def _answer(q: str) -> str:
        return f"echo: {q}"

    @app.post("/chat")
    async def chat(payload: dict):
        # produces at least one child span under the fastapi.request span
        return {"answer": _answer(payload.get("q", ""))}

    # --- routes used only by test_concurrent_requests_get_isolated_spans below ---
    # /chat_a deliberately holds its request open (after creating its own child
    # span) until /release_a is hit from the test, so it is genuinely still
    # in-flight -- its span sitting unclaimed in the shared trace-return buffer
    # (TraceReturnSpanExporter) -- while /chat_b runs an entire separate
    # request end-to-end, including popping its own trace's spans. This
    # deterministically (no sleeps) forces both requests' tagged spans to sit
    # in the shared buffer at the same time, which is exactly the window in
    # which a broken trace_id filter in pop_spans_for_trace would leak spans
    # across requests. /wait_for_a_ready and /release_a are pure
    # test-orchestration endpoints -- not part of the trace-return feature.
    a_ready = asyncio.Event()
    release_a_event = asyncio.Event()

    @monocle_trace_method(span_name="answer_question_a")
    def _answer_a(q: str) -> str:
        return f"echo-a: {q}"

    @monocle_trace_method(span_name="answer_question_b")
    def _answer_b(q: str) -> str:
        return f"echo-b: {q}"

    @app.post("/chat_a")
    async def chat_a(payload: dict):
        result = _answer_a(payload.get("q", ""))
        a_ready.set()
        await asyncio.wait_for(release_a_event.wait(), timeout=10)
        return {"answer": result}

    @app.post("/chat_b")
    async def chat_b(payload: dict):
        return {"answer": _answer_b(payload.get("q", ""))}

    @app.post("/wait_for_a_ready")
    async def wait_for_a_ready():
        await asyncio.wait_for(a_ready.wait(), timeout=10)
        return {}

    @app.post("/release_a")
    async def release_a():
        release_a_event.set()
        return {}

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


def test_server_spans_returned_in_band(server):
    """The FastAPI server's spans are piggybacked on the response and land in the validator.

    The x-monocle-retrieve-traces header is NOT passed explicitly here: with
    MONOCLE_TRACE_RETRIEVAL_KEY set at module import time (above), HttpRunner
    (_maybe_inject_retrieval_key) auto-injects the header, and the server's
    default_trace_retrieval_callback authorizes it against
    MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY -- proving the key-based auto-injection
    path end-to-end, not just the header-based mechanics.
    """
    import json
    from monocle_test_tools.validator import MonocleValidator

    validator = MonocleValidator()
    response = validator.run_agent(
        server + "/chat", "http",
        method="POST", json={"q": "hi"},
    )
    # (a) body is clean -- trailer was stripped
    assert response.json()["answer"] == "echo: hi"
    # (b) DISCRIMINATING: spans arrived in-band via the response trailer, not just
    # via the shared in-process TracerProvider/memory_exporter (which the FastAPI
    # server and the validator both happen to share in this in-process test setup,
    # and which would make the assertion below pass regardless of whether the
    # piggyback feature actually works). `_monocle_remote_spans` is set ONLY by the
    # client-side RequestSpanHandler.post_task_processing when the server emitted
    # the `x-monocle-traces` trailer -- it is the true evidence spans traveled back
    # in-band on the HTTP response.
    raw = getattr(response, "_monocle_remote_spans", None)
    assert raw is not None, "server did not piggyback spans on the HTTP response (no _monocle_remote_spans)"
    remote_names = [s.get("name") for s in json.loads(raw)]
    assert "answer_question" in remote_names, f"answer_question not in piggybacked spans: {remote_names}"
    # (c) additional, non-load-bearing: the deserialized remote span also ends up
    # merged into validator.spans via HttpRunner/add_remote_spans.
    span_names = [s.name for s in validator.spans]
    assert "answer_question" in span_names


def test_wrong_key_denied(server):
    """A request that presents the wrong trace-retrieval key is denied end-to-end.

    Discriminating counterpart to test_server_spans_returned_in_band: proves
    that merely sending the x-monocle-retrieve-traces header is not enough --
    the value must match the server's MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY, or
    the server withholds the trailer entirely (no x-monocle-traces header,
    clean body).
    """
    import requests

    r = requests.post(
        server + "/chat", json={"q": "x"},
        headers={"x-monocle-retrieve-traces": "not-the-key"},
    )
    assert r.json()["answer"] == "echo: x"  # body clean
    assert "x-monocle-traces" not in {k.lower() for k in r.headers.keys()}  # no trailer header


def test_concurrent_requests_get_isolated_spans(server):
    """Two overlapping opted-in requests each get back ONLY their own trace's spans.

    This proves the SERVER-side trace_id eviction (TraceReturnSpanExporter.
    pop_spans_for_trace, called from HttpSpanHandler.build_trace_return_trailer)
    actually filters by trace_id instead of e.g. draining the whole shared
    buffer. /chat_a and /chat_b are two distinctly-named
    @monocle_trace_method-decorated routes (see the `server` fixture above).
    /chat_a is held open by the server (via asyncio.Event, not a sleep) after
    its own span is created, so it is provably still in-flight -- with its
    tagged span sitting unclaimed in the shared buffer -- while /chat_b runs an
    entire separate request/response cycle, including popping its own trace's
    spans. Only then is /chat_a released.

    Driving this via two concurrent `MonocleValidator().run_agent(...)` calls
    would race on `RequestSpanHandler._trace_all_urls`, a shared class-level
    flag that HttpRunner flips True/False around each call in its own
    try/finally -- with two threads doing that concurrently, one thread's
    `finally` can flip tracing off while the other request is still in flight,
    which would silently defeat the client-side trailer-stripping this test
    also relies on. So, per the task's suggested fallback, this drives the two
    requests directly via `requests` with `RequestSpanHandler.
    set_trace_all_urls_for_test(True)` held for the whole concurrent exchange
    (the same mechanism HttpRunner itself uses) -- avoiding that race while
    still exercising genuine server-side concurrency, which is the property
    under test.
    """
    import concurrent.futures
    import json

    import requests
    from monocle_apptrace.instrumentation.metamodel.requests._helper import RequestSpanHandler
    from monocle_test_tools.validator import MonocleValidator

    # Ensure setup_monocle_telemetry() has run before we touch the fastapi/
    # requests instrumentation directly. MonocleValidator() is a process-wide
    # singleton (see validator.py) and its __init__ is a no-op after the first
    # call, so constructing it here is safe and order-independent -- this test
    # does not otherwise depend on test_server_spans_returned_in_band having
    # run first to have already triggered setup.
    MonocleValidator()

    # Value must match MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY (set at module
    # import time above) -- these requests go via raw `requests`, not
    # HttpRunner, so there is no auto-injection here; the key must be
    # supplied explicitly to be authorized under the key-based gate.
    headers = {"x-monocle-retrieve-traces": os.environ["MONOCLE_TRACE_RETRIEVAL_KEY"]}
    RequestSpanHandler.set_trace_all_urls_for_test(True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(
                requests.post, server + "/chat_a", json={"q": "a"}, headers=headers, timeout=15
            )
            # Block (deterministically, not via a sleep) until /chat_a's route
            # has actually created its span and reached its hold point.
            requests.post(server + "/wait_for_a_ready", timeout=15).raise_for_status()

            # /chat_a is now genuinely in-flight: its span is tagged in the
            # shared trace-return buffer, unclaimed. Run /chat_b fully through
            # to completion while that's true.
            response_b = requests.post(server + "/chat_b", json={"q": "b"}, headers=headers, timeout=15)
            response_b.raise_for_status()

            # /chat_b has now popped its own trace's spans. Release /chat_a.
            requests.post(server + "/release_a", timeout=15).raise_for_status()

            response_a = future_a.result(timeout=15)
            response_a.raise_for_status()
    finally:
        RequestSpanHandler.set_trace_all_urls_for_test(False)

    raw_b = getattr(response_b, "_monocle_remote_spans", None)
    assert raw_b is not None, "/chat_b did not receive piggybacked spans"
    names_b = [s.get("name") for s in json.loads(raw_b)]
    assert "answer_question_b" in names_b, f"answer_question_b missing from /chat_b spans: {names_b}"
    assert "answer_question_a" not in names_b, (
        f"trace_id isolation broken: /chat_b leaked /chat_a's still in-flight span: {names_b}"
    )

    raw_a = getattr(response_a, "_monocle_remote_spans", None)
    assert raw_a is not None, "/chat_a did not receive piggybacked spans"
    names_a = [s.get("name") for s in json.loads(raw_a)]
    assert "answer_question_a" in names_a, f"answer_question_a missing from /chat_a spans: {names_a}"
    assert "answer_question_b" not in names_a, (
        f"trace_id isolation broken: /chat_a leaked /chat_b's already-popped span: {names_a}"
    )


def test_inert_when_disabled(server):
    os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "false"
    try:
        import requests

        r = requests.post(server + "/chat", json={"q": "x"},
                          headers={"x-monocle-retrieve-traces": "true"})
        assert "x-monocle-traces" not in {k.lower() for k in r.headers.keys()}
        assert r.json()["answer"] == "echo: x"
    finally:
        os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "true"
