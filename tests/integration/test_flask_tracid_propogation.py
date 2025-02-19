import pytest
from threading import Thread
import os, time
from flask import Flask, request, jsonify
import requests, uuid
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from common import flask_helper
from common.custom_exporter import CustomConsoleSpanExporter
from common.chain_exec import TestScopes, setup_chain
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry, start_scope, stop_scope

CHAT_SCOPE_NAME = "chat"
CONVERSATION_SCOPE_NAME = "discussion"
CONVERSATION_SCOPE_VALUE = "conv1234"
custom_exporter = CustomConsoleSpanExporter()


@pytest.fixture(scope="session")
def setup():
    flask_helper.start_flask()
    setup_monocle_telemetry(workflow_name = "flask_test", span_processors=[SimpleSpanProcessor(custom_exporter)])
    yield None
    flask_helper.stop_flask()

@pytest.mark.integration()
def test(setup):
    custom_exporter.reset()
    client_session_id = f"{uuid.uuid4().hex}"
    prompt = "What is Task Decomposition?"
    headers = {"client-id": client_session_id}
    url = flask_helper.get_url()
    token = start_scope(CONVERSATION_SCOPE_NAME, CONVERSATION_SCOPE_VALUE)
    response = requests.get(f"{url}/chat?question={prompt}", headers=headers)
    print (response)
    stop_scope(CONVERSATION_SCOPE_NAME, token)

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
        if trace_id is None:
            trace_id = span.context.trace_id
        else:
            assert trace_id == span.context.trace_id

