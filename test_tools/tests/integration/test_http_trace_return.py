import os

# MONOCLE_ENABLE_TRACE_RETURN must be "true" *before* MonocleValidator() is
# constructed anywhere in this process: setup_monocle_telemetry() only wires
# up the TraceReturnSpanExporter processor once, at instrumentor-setup time,
# and MonocleValidator is a session-wide singleton. Setting it here at module
# import time guarantees it is in place before the first construction, as
# long as this file is run on its own / is the first test module to touch
# MonocleValidator in the pytest session.
os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "true"

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
    """The FastAPI server's spans are piggybacked on the response and land in the validator."""
    import json
    from monocle_test_tools.validator import MonocleValidator

    validator = MonocleValidator()
    response = validator.run_agent(
        server + "/chat", "http",
        method="POST", json={"q": "hi"},
        headers={"x-monocle-retrieve-traces": "true"},
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
