"""End-to-end trace-return coverage for the non-FastAPI HTTP server frameworks:
Flask, aiohttp (Task 6 of the "trace-return across all HTTP frameworks" feature).

Mirrors test_http_trace_return.py (the FastAPI e2e): the env vars below must be
set *before* MonocleValidator() / setup_monocle_telemetry() is first constructed
anywhere in the process, because the trace-return SimpleSpanProcessor is wired
up once, at instrumentor-setup time (see _append_trace_return_processor in
instrumentor.py). Setting them here at module import time guarantees that, as
long as this file is the first test module in the pytest session to touch
MonocleValidator (true when run standalone, per the task's run command).
"""
import os

os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "true"
os.environ["MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY"] = "fw-s3cret"
os.environ["MONOCLE_TRACE_RETRIEVAL_KEY"] = "fw-s3cret"

import asyncio
import json
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


# ---------------------------------------------------------------------------
# Flask (WSGI, served by werkzeug)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flask_server():
    flask = pytest.importorskip("flask")
    from werkzeug.serving import make_server
    from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method

    app = flask.Flask(__name__)

    @monocle_trace_method(span_name="answer_question")
    def _answer(q):
        return f"echo: {q}"

    @app.post("/chat")
    def chat():
        data = flask.request.get_json(force=True, silent=True) or {}
        return {"answer": _answer(data.get("q", ""))}

    port = _free_port()
    srv = make_server("127.0.0.1", port, app)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    t.join(timeout=5)


def test_flask_server_spans_returned(flask_server, monocle_trace_asserter):
    """DISCRIMINATING: proves the Flask server piggybacked its child span on the
    HTTP response trailer (`_monocle_remote_spans`, set only by the client-side
    RequestSpanHandler.post_task_processing when the server emitted the
    x-monocle-traces trailer) -- not merely that the (shared, in-process)
    TracerProvider/memory_exporter happens to also see the span, which would
    make the assertion pass regardless of whether the piggyback actually works.
    """
    response = monocle_trace_asserter.run_agent(
        flask_server + "/chat", "http", method="POST", json={"q": "hi"})
    assert response.json()["answer"] == "echo: hi"
    raw = getattr(response, "_monocle_remote_spans", None)
    assert raw is not None, "no piggybacked spans from Flask server"
    names = [s.get("name") for s in json.loads(raw)]
    assert "answer_question" in names, f"answer_question not in piggybacked spans: {names}"


def test_flask_wrong_key_denied(flask_server):
    """Discriminating counterpart: a request presenting the wrong retrieval key
    gets no trailer at all (clean body, no x-monocle-traces header) -- merely
    sending the header is not enough, the value must match."""
    import requests

    r = requests.post(
        flask_server + "/chat", json={"q": "x"},
        headers={"x-monocle-retrieve-traces": "not-the-key"},
    )
    assert r.json()["answer"] == "echo: x"
    assert "x-monocle-traces" not in {k.lower() for k in r.headers.keys()}


def test_flask_inert_when_disabled(flask_server):
    os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "false"
    try:
        import requests

        r = requests.post(flask_server + "/chat", json={"q": "x"},
                           headers={"x-monocle-retrieve-traces": os.environ["MONOCLE_TRACE_RETRIEVAL_KEY"]})
        assert "x-monocle-traces" not in {k.lower() for k in r.headers.keys()}
        assert r.json()["answer"] == "echo: x"
    finally:
        os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "true"


# ---------------------------------------------------------------------------
# aiohttp (asyncio, served by aiohttp.web in a background thread/event loop)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aiohttp_server():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method

    @monocle_trace_method(span_name="answer_question")
    def _answer(q):
        return f"echo: {q}"

    async def chat(request):
        data = await request.json()
        return web.json_response({"answer": _answer(data.get("q", ""))})

    app = web.Application()
    app.router.add_post("/chat", chat)

    port = _free_port()
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    state = {}

    def _run_loop():
        asyncio.set_event_loop(loop)

        async def _start():
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", port)
            await site.start()
            state["runner"] = runner

        loop.run_until_complete(_start())
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    assert ready.wait(timeout=5), "aiohttp server thread did not start in time"

    yield f"http://127.0.0.1:{port}"

    fut = asyncio.run_coroutine_threadsafe(state["runner"].cleanup(), loop)
    fut.result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


def test_aiohttp_server_spans_returned(aiohttp_server, monocle_trace_asserter):
    """DISCRIMINATING, mirrors test_flask_server_spans_returned: the aiohttp
    server's child span arrives via the response trailer (_monocle_remote_spans),
    proving the buffered web.Response injection path (aiohttp _helper.py's
    _aiohttp_inject_buffered, wired into aiohttpSpanHandler.post_task_processing)
    works end-to-end through a real running server. The streaming
    (StreamResponse prepare/write_eof) path is unit-covered by
    apptrace/tests/unit/test_aiohttp_trace_return.py (Task 5) and is not
    re-exercised here since it needs no real server to prove correct."""
    response = monocle_trace_asserter.run_agent(
        aiohttp_server + "/chat", "http", method="POST", json={"q": "hi"})
    assert response.json()["answer"] == "echo: hi"
    raw = getattr(response, "_monocle_remote_spans", None)
    assert raw is not None, "no piggybacked spans from aiohttp server"
    names = [s.get("name") for s in json.loads(raw)]
    assert "answer_question" in names, f"answer_question not in piggybacked spans: {names}"


def test_aiohttp_wrong_key_denied(aiohttp_server):
    import requests

    r = requests.post(
        aiohttp_server + "/chat", json={"q": "x"},
        headers={"x-monocle-retrieve-traces": "not-the-key"},
    )
    assert r.json()["answer"] == "echo: x"
    assert "x-monocle-traces" not in {k.lower() for k in r.headers.keys()}


# ---------------------------------------------------------------------------
# AWS Lambda -- wrapper-level (no real server / AWS runtime): drive the route
# decorator directly with a fake event/context, mirroring how AWS itself would
# invoke a decorated handler.
# ---------------------------------------------------------------------------

def test_lambda_route_wrapper_returns_trailer():
    from monocle_test_tools.validator import MonocleValidator
    from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method
    from monocle_apptrace.instrumentation.metamodel.lambdafunc.wrapper import (
        monocle_trace_lambda_function_route,
    )
    from monocle_apptrace.instrumentation.common import trace_return as tr

    # Ensures setup_monocle_telemetry() has run (idempotent singleton) with the
    # trace-return processor wired up, per the module-level env vars above.
    MonocleValidator()

    @monocle_trace_method(span_name="answer_question")
    def _answer(q):
        return f"echo: {q}"

    @monocle_trace_lambda_function_route
    def handler(event, context):
        body = json.loads(event.get("body") or "{}")
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"answer": _answer(body.get("q", ""))}),
        }

    fake_event = {
        "headers": {"x-monocle-retrieve-traces": "fw-s3cret"},
        "body": json.dumps({"q": "hi"}),
        "httpMethod": "POST",
        "path": "/chat",
        "requestContext": {},
        "queryStringParameters": {},
    }

    # Called with event/context as KEYWORD args: lambdaSpanHandler.pre_tracing
    # (lambda_func_pre_tracing) reads kwargs['event'] to extract headers and
    # authorize/tag the trace-return scope before the child span is created --
    # a positional call leaves kwargs empty and the scope never gets set
    # (verified empirically against the current lambdafunc/_helper.py).
    result = handler(event=fake_event, context=None)

    assert result["statusCode"] == 200
    headers = result.get("headers", {})
    header_value = next((v for k, v in headers.items() if k.lower() == "x-monocle-traces"), None)
    assert header_value is not None, "lambda route wrapper did not return the trace-return trailer header"

    delim = tr.parse_delimiter_from_header(header_value)
    clean_body, payload = tr.split_body_and_trailer(result["body"].encode("utf-8"), delim)
    # (a) body a client would see (after stripping) is clean
    assert json.loads(clean_body.decode())["answer"] == "echo: hi"
    # (b) the client-side strip recovers the actual span JSON
    spans = json.loads(tr.decode_payload(payload))
    names = [s.get("name") for s in spans]
    assert "answer_question" in names, f"answer_question not in recovered lambda spans: {names}"


def test_lambda_route_wrapper_wrong_key_denied():
    from monocle_test_tools.validator import MonocleValidator
    from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method
    from monocle_apptrace.instrumentation.metamodel.lambdafunc.wrapper import (
        monocle_trace_lambda_function_route,
    )

    MonocleValidator()

    @monocle_trace_method(span_name="answer_question_wrong_key")
    def _answer(q):
        return f"echo: {q}"

    @monocle_trace_lambda_function_route
    def handler(event, context):
        body = json.loads(event.get("body") or "{}")
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"answer": _answer(body.get("q", ""))}),
        }

    fake_event = {
        "headers": {"x-monocle-retrieve-traces": "not-the-key"},
        "body": json.dumps({"q": "hi"}),
        "httpMethod": "POST",
        "path": "/chat",
        "requestContext": {},
        "queryStringParameters": {},
    }
    result = handler(event=fake_event, context=None)
    headers = result.get("headers", {})
    assert not any(k.lower() == "x-monocle-traces" for k in headers), (
        "lambda route wrapper returned a trailer despite the wrong retrieval key"
    )
    assert json.loads(result["body"])["answer"] == "echo: hi"


# ---------------------------------------------------------------------------
# Azure Functions -- wrapper-level (no real Functions host): drive the route
# decorator directly with a real azure.functions.HttpRequest/HttpResponse.
# ---------------------------------------------------------------------------

def test_azure_function_route_wrapper_returns_trailer():
    azure_functions = pytest.importorskip("azure.functions")
    from monocle_test_tools.validator import MonocleValidator
    from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method
    from monocle_apptrace.instrumentation.metamodel.azfunc.wrapper import (
        monocle_trace_azure_function_route,
    )
    from monocle_apptrace.instrumentation.common import trace_return as tr

    MonocleValidator()

    @monocle_trace_method(span_name="answer_question")
    def _answer(q):
        return f"echo: {q}"

    @monocle_trace_azure_function_route
    def handler(req, context=None):
        raw = req.get_body()
        body = json.loads(raw.decode("utf-8")) if raw else {}
        return azure_functions.HttpResponse(
            body=json.dumps({"answer": _answer(body.get("q", ""))}).encode("utf-8"),
            status_code=200,
            mimetype="application/json",
        )

    fake_req = azure_functions.HttpRequest(
        method="POST",
        url="http://localhost/chat",
        headers={"x-monocle-retrieve-traces": "fw-s3cret"},
        params={},
        route_params={},
        body=json.dumps({"q": "hi"}).encode("utf-8"),
    )

    # Called with req/context as KEYWORD args, mirroring the Lambda case:
    # azureSpanHandler.pre_tracing (azure_func_pre_tracing) reads kwargs['req'].
    result = handler(req=fake_req, context=None)

    header_value = result.headers.get("x-monocle-traces")
    assert header_value is not None, "azure function route wrapper did not return the trace-return trailer header"

    delim = tr.parse_delimiter_from_header(header_value)
    clean_body, payload = tr.split_body_and_trailer(result.get_body(), delim)
    assert json.loads(clean_body.decode())["answer"] == "echo: hi"
    spans = json.loads(tr.decode_payload(payload))
    names = [s.get("name") for s in spans]
    assert "answer_question" in names, f"answer_question not in recovered azfunc spans: {names}"


def test_azure_function_route_wrapper_wrong_key_denied():
    azure_functions = pytest.importorskip("azure.functions")
    from monocle_test_tools.validator import MonocleValidator
    from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method
    from monocle_apptrace.instrumentation.metamodel.azfunc.wrapper import (
        monocle_trace_azure_function_route,
    )

    MonocleValidator()

    @monocle_trace_method(span_name="answer_question_wrong_key")
    def _answer(q):
        return f"echo: {q}"

    @monocle_trace_azure_function_route
    def handler(req, context=None):
        raw = req.get_body()
        body = json.loads(raw.decode("utf-8")) if raw else {}
        return azure_functions.HttpResponse(
            body=json.dumps({"answer": _answer(body.get("q", ""))}).encode("utf-8"),
            status_code=200,
            mimetype="application/json",
        )

    fake_req = azure_functions.HttpRequest(
        method="POST",
        url="http://localhost/chat",
        headers={"x-monocle-retrieve-traces": "not-the-key"},
        params={},
        route_params={},
        body=json.dumps({"q": "hi"}).encode("utf-8"),
    )
    result = handler(req=fake_req, context=None)
    assert result.headers.get("x-monocle-traces") is None, (
        "azure function route wrapper returned a trailer despite the wrong retrieval key"
    )
    assert json.loads(result.get_body().decode())["answer"] == "echo: hi"
