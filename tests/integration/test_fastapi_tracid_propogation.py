import pytest
import uuid
import os
import requests
from common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry, start_scope, stop_scope
from monocle_apptrace.instrumentation.common.constants import SCOPE_METHOD_FILE, SCOPE_CONFIG_PATH, TRACE_PROPOGATION_URLS
from tests.common import fastapi_helper

custom_exporter = CustomConsoleSpanExporter()
CHAT_SCOPE_NAME = "chat"
CONVERSATION_SCOPE_NAME = "discussion"
CONVERSATION_SCOPE_VALUE = "conv1234"

@pytest.fixture(scope="module")
def setup():
    print("Setting up FastAPI server")
    os.environ[TRACE_PROPOGATION_URLS] = "http://127.0.0.1"
    os.environ[SCOPE_CONFIG_PATH] = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCOPE_METHOD_FILE)
    fastapi_helper.start_fastapi()
    setup_monocle_telemetry(
        workflow_name="fastapi_test",
        span_processors=[SimpleSpanProcessor(custom_exporter)]
    )

@pytest.fixture(autouse=True)
def pre_test():
    custom_exporter.reset()

@pytest.mark.integration()
def test_chat_endpoint(setup):
    custom_exporter.reset()
    client_session_id = f"{uuid.uuid4().hex}"
    headers = {"client-id": client_session_id}
    url = fastapi_helper.get_url()
    question = "What is Task Decomposition?"
    token = start_scope(CONVERSATION_SCOPE_NAME, CONVERSATION_SCOPE_VALUE)
    resp = requests.get(f"{url}/chat?question={question}", headers=headers, data={"test": "123"})
    print(f"Response status: {resp.status_code}")
    print(f"Response text: {resp.text}")
    assert resp.status_code == 200
    stop_scope(token)
    verify_scopes()

def verify_scopes():
    scope_name = "conversation"
    spans = custom_exporter.get_captured_spans()
    message_scope_id = None
    trace_id = None
    for span in spans:
        span_attributes = span.attributes
        if span_attributes.get("span.type", "") in ["inference", "retrieval"]:
            if message_scope_id is None:
                message_scope_id = span_attributes.get("scope."+scope_name)
                assert message_scope_id is not None
            else:
                assert message_scope_id == span_attributes.get("scope."+scope_name)
        if span_attributes.get("span.type", "") == "http.send":
            span_input, span_output = span.events
            assert span_attributes.get("entity.1.method").lower() == "get"
            assert span_attributes.get("entity.1.URL") is not None
            assert span_output.attributes['status'] == "200"
        if span_attributes.get("span.type", "") == "http.process":
            assert span_attributes.get("entity.1.method").lower() == "get"
            assert span_attributes.get("entity.1.route") is not None
        if trace_id is None:
            trace_id = span.context.trace_id
        else:
            assert trace_id == span.context.trace_id