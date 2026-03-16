import os
import socket
import threading
import time
import pytest
import requests
from dotenv import load_dotenv
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from common.msagent_server import start_msgent_server

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_DEPLOYMENT = os.getenv("AZURE_OPENAI_API_DEPLOYMENT")


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture(scope="module")
def setup():
    """Set up and tear down Monocle telemetry instrumentation."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter)
    ]
    host = "127.0.0.1"
    port = 8088

    if not _is_port_open(host, port):
        os.environ["HOSTED_AGENT_MODE"] = "true"
        thread = threading.Thread(target=start_msgent_server, daemon=True)
        thread.start()

        start = time.time()
        while time.time() - start < 30:
            if _is_port_open(host, port):
                break
            if not thread.is_alive():
                pytest.skip("msagent_server thread exited before becoming ready")
            time.sleep(0.5)
        else:
            pytest.skip("msagent_server did not become ready on http://localhost:8088")
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="microsoft_agent_simple_test",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
    


def test_msagent_stream_requests_call(setup):
    """Call local hosted agent endpoint exposed by msagent_server.py."""
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_DEPLOYMENT"):
        pytest.skip("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_DEPLOYMENT are required")

    try:
        response = requests.post(
            "http://localhost:8088/responses",
            headers={"Content-Type": "application/json"},
            json={
                "input": "Book a flight from BOM to JFK for January 15th, 2026.",
                "stream": True,
            },
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        pytest.skip(
            "Hosted adapter is not running at http://localhost:8088. "
            "Start it with: python apptrace/tests/integration/msagent_server.py --hosted"
        )

    assert response.status_code == 200, response.text
    body = response.text.strip()
    assert body, "endpoint returned empty response"
    assert "Invalid JSON payload" not in body, body
    assert "FLIGHT BOOKING CONFIRMED" in body or "data:" in body, body
    
    verify_spans(setup)
    
    
def verify_spans(custom_exporter):
    time.sleep(2)
    spans = custom_exporter.get_captured_spans()

    for span in spans:
        if span.name  == "agent_framework.ChatAgent.run_stream":
            output_event = None
            for event in span.events:
                if event.name == "data.output":
                    output_event = event
                    break
            
            assert output_event is not None, "data.output event should exist"
            response_data = output_event.attributes.get("response", None)
            assert response_data is not None, "response_data should exist"
            
        if span.name  == "agent_framework.azure._assistants_client.AzureOpenAIAssistantsClient.get_streaming_response":
            output_event = None
            for event in span.events:
                if event.name == "data.output":
                    output_event = event
                    break
            
            assert output_event is not None, "data.output event should exist"
            response_data = output_event.attributes.get("response", None)
            assert response_data is not None, "response_data should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
    


