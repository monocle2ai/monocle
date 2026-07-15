import logging
import multiprocessing
import time
from typing import Optional
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


import requests

from monocle_apptrace.exporters.okahu.okahu_exporter import OkahuSpanExporter

PORT = 8096
CONVERSATION_SCOPE_NAME = "discussion"
CONVERSATION_SCOPE_VALUE = "conv1234"

_server_process: Optional[multiprocessing.Process] = None
logger = logging.getLogger(__name__)


def _run_server(port: int = PORT) -> None:
    """Subprocess entry point.

    All heavy imports and Monocle telemetry setup happen *inside* this function
    so they only run in the dedicated server process and don't pollute the
    test (parent) process.
    """
    import uvicorn
    from fastapi import FastAPI, Request

    from monocle_apptrace import setup_monocle_telemetry
    from test_common.adk_travel_agent import run_agent

    setup_monocle_telemetry(
        workflow_name="okahu_test_fastapi_service",
        span_processors=[SimpleSpanProcessor(OkahuSpanExporter())]
    )

    app = FastAPI()
    server_logger = logging.getLogger("fastapi_helper.server")

    @app.post("/api/v1/ask_agent")
    async def ask_agent(request: Request):
        try:
            body = await request.json()
            question = body.get("query", "")
            response = await run_agent(question)
            return {"answer": response}
        except Exception as e:
            server_logger.error(f"Error in ask_agent: {e}")
            return {"Status": "Failure --- some error occurred"}

    @app.get("/")
    def health_check():
        return {}

    @app.get("/hello")
    def hello():
        return {"Status": "Success"}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def start_fastapi() -> None:
    """Launch the FastAPI server in a separate process using the 'spawn' start
    method. Spawn ensures the child re-imports modules from scratch so the
    parent process is never touched by Monocle telemetry / ADK initialization.
    """
    global _server_process
    if _server_process is not None and _server_process.is_alive():
        logger.info("FastAPI server already running (pid=%s)", _server_process.pid)
        return
    ctx = multiprocessing.get_context("spawn")
    _server_process = ctx.Process(
        target=_run_server,
        args=(PORT,),
        name="fastapi-test-server",
        daemon=True,
    )
    _server_process.start()
    logger.info("Started FastAPI server subprocess (pid=%s)", _server_process.pid)


def stop_fastapi(timeout: float = 10.0) -> None:
    """Terminate the FastAPI server subprocess started by start_fastapi()."""
    global _server_process
    if _server_process is None:
        return
    if _server_process.is_alive():
        logger.info("Terminating FastAPI server subprocess (pid=%s)", _server_process.pid)
        _server_process.terminate()
        _server_process.join(timeout=timeout)
        if _server_process.is_alive():
            logger.warning(
                "FastAPI server subprocess (pid=%s) did not exit after terminate; killing",
                _server_process.pid,
            )
            _server_process.kill()
            _server_process.join(timeout=timeout)
    _server_process = None
    logger.info("FastAPI server stopped")


def wait_for_server(timeout: float = 30.0, interval: float = 1.0) -> bool:
    """Poll the FastAPI health check endpoint until it returns 200.

    Args:
        timeout: Maximum number of seconds to wait for a healthy response.
        interval: Seconds between successive health check attempts.

    Returns:
        True once the server responds with 200, otherwise raises RuntimeError.
    """
    health_url = get_url() + "/hello"
    deadline = time.monotonic() + timeout
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(health_url, timeout=interval)
            if response.status_code == 200:
                logger.info("FastAPI server is up at %s", get_url())
                return True
            last_error = RuntimeError(f"health check returned {response.status_code}")
        except Exception as exc:  # noqa: BLE001 — surface any connection-level error
            last_error = exc
        time.sleep(interval)
    raise RuntimeError(
        f"FastAPI server at {health_url} did not become healthy within {timeout}s: {last_error}"
    )


def get_url() -> str:
    return f"http://127.0.0.1:{PORT}"


if __name__ == "__main__":
    start_fastapi()
    try:
        wait_for_server()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_fastapi()
