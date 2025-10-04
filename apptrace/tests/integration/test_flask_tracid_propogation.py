import os
import uuid

import pytest
import requests
from common import flask_helper
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace import setup_monocle_telemetry, start_scope, stop_scope
from monocle_apptrace.instrumentation.common.constants import (
    SCOPE_CONFIG_PATH,
    SCOPE_METHOD_FILE,
    TRACE_PROPOGATION_URLS,
)
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

CHAT_SCOPE_NAME = "chat"
CONVERSATION_SCOPE_NAME = "discussion"
CONVERSATION_SCOPE_VALUE = "conv1234"
custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="function", autouse=True)
def setup():
    try:
        print ("Setting up Flask")
        os.environ[TRACE_PROPOGATION_URLS] = "http://127.0.0.1"
        os.environ[SCOPE_CONFIG_PATH] = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCOPE_METHOD_FILE)
        flask_helper.start_flask()
        instrumentor = setup_monocle_telemetry(workflow_name = "flask_test", span_processors=[SimpleSpanProcessor(custom_exporter)])
        yield
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

@pytest.fixture(autouse=True)
def pre_test():
    # clear old spans
   custom_exporter.reset()

def test_http_flask_scope(setup):
    custom_exporter.reset()
    client_session_id = f"{uuid.uuid4().hex}"
    prompt = "What is Task Decomposition?"
    headers = {"client-id": client_session_id}
    url = flask_helper.get_url()
    token = start_scope(CONVERSATION_SCOPE_NAME, CONVERSATION_SCOPE_VALUE)
    response = requests.get(f"{url}/chat?question={prompt}", headers=headers, data={"test": "123"})
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
            assert span_attributes.get("scope."+CONVERSATION_SCOPE_NAME) == CONVERSATION_SCOPE_VALUE
        if span_attributes.get("span.type", "") == "http.send":
            span_input, span_output = span.events
            assert span_attributes.get("entity.1.method").lower() == "get"
            assert span_attributes.get("entity.1.URL") is not None
            assert span_output.attributes['status'] == "200"
        if span_attributes.get("span.type", "") == "http.process":
            #span_input, span_output = span.events
            assert span_attributes.get("entity.1.method").lower() == "get"
            assert span_attributes.get("entity.1.route") is not None
            assert span_attributes.get("entity.1.url") is not None
            #assert span_output.attributes['status'] == "200"
        if trace_id is None:
            trace_id = span.context.trace_id
        else:
            temp_trace_id = None
            if temp_trace_id is None or span.name.startswith("requests.sessions") or temp_trace_id == span.context.trace_id:
                temp_trace_id = span.context.trace_id
            else:
                assert trace_id == span.context.trace_id

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
