import logging
import os
import uuid

import pytest
import requests
from common import fastapi_helper
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace import setup_monocle_telemetry, start_scope, stop_scope
from monocle_apptrace.instrumentation.common.constants import (
    SCOPE_CONFIG_PATH,
    SCOPE_METHOD_FILE,
    TRACE_PROPOGATION_URLS,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from monocle_apptrace.exporters.file_exporter import FileSpanExporter

CHAT_SCOPE_NAME = "chat"
CONVERSATION_SCOPE_NAME = "discussion"
CONVERSATION_SCOPE_VALUE = "conv1234"

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module", autouse=True)
def setup():
    try:
        os.environ["USER_AGENT"] = "monocle-apptrace-tests"
        logger.info("Setting up FastAPI server")
        os.environ[TRACE_PROPOGATION_URLS] = "http://127.0.0.1"
        os.environ[SCOPE_CONFIG_PATH] = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCOPE_METHOD_FILE)
        custom_exporter = CustomConsoleSpanExporter()
        fastapi_helper.start_fastapi()
        import time
        time.sleep(5)  # Give server time to start
        instrumentor = setup_monocle_telemetry(
            workflow_name="fastapi_test",
            monocle_exporters_list="file"
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
    fastapi_helper.stop_fastapi()


def test_chat_endpoint(setup):
    client_session_id = f"{uuid.uuid4().hex}"
    headers = {"client-id": client_session_id}
    url = "http://127.0.0.1:8096"
    question = "What is Task Decomposition?"
    token = start_scope(CONVERSATION_SCOPE_NAME, CONVERSATION_SCOPE_VALUE)
    resp = requests.get(f"{url}/chat?question={question}", headers=headers, data={"test": "123"})
    logger.info(f"Response status: {resp.status_code}")
    logger.info(f"Response text: {resp.text}")
    assert resp.status_code == 200
    stop_scope(token)
    verify_scopes(setup)

def verify_scopes(setup):
    spans = setup.get_captured_spans()
    logger.info(f"Total spans captured: {len(spans)}")
    message_scope_id = None
    trace_id = None
    for span in spans:
        span_attributes = span.attributes
        logger.info(f"Span type: {span_attributes.get('span.type', 'unknown')}")
        logger.info(f"Span attributes: {dict(span_attributes)}")
        if span_attributes.get("span.type", "") in ["inference", "retrieval"]:
            if message_scope_id is None:
                message_scope_id = span_attributes.get("scope."+CONVERSATION_SCOPE_NAME)
                logger.info(f"Found scope.{CONVERSATION_SCOPE_NAME}: {message_scope_id}")
                assert message_scope_id is not None, f"No scope.{CONVERSATION_SCOPE_NAME} found in span attributes: {dict(span_attributes)}"
            else:
                assert message_scope_id == span_attributes.get("scope."+CONVERSATION_SCOPE_NAME)
        
        # Verify FastAPI response spans specifically
        if span.name == "fastapi.response":
            assert span_attributes.get("span.type", "") == "http.send"
            logger.info("Found FastAPI response span with correct span.type: http.send")
            
            # Verify the span has the expected events
            if len(span.events) == 2:
                span_input, span_output = span.events
                assert span_input.name == "data.input", f"Expected input event name 'data.input', got '{span_input.name}'"
                assert span_output.name == "data.output", f"Expected output event name 'data.output', got '{span_output.name}'"
                
                # Check that we have error_code in output
                if 'error_code' in span_output.attributes:
                    assert span_output.attributes['error_code'] == "200", f"Expected error_code '200', got '{span_output.attributes['error_code']}'"
                    logger.info("FastAPI response span has correct error_code: 200")

        if span_attributes.get("span.type", "") == "http.send":
            if len(span.events) == 2:
                span_input, span_output = span.events
                method = span_attributes.get("entity.1.method")
                if method is not None:
                    assert method.lower() == "get"
                # Check status in output if it exists
                if 'status' in span_output.attributes:
                    assert span_output.attributes['status'] == "200"
        if span_attributes.get("span.type", "") == "http.process":
            method = span_attributes.get("entity.1.method")
            if method is not None:
                assert method.lower() == "get"
            # Note: Some spans may not have entity.1.route or entity.1.url attributes
        if trace_id is None:
            trace_id = span.context.trace_id
        # else:
        #     assert trace_id == span.context.trace_id

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
