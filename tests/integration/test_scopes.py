import os
import bs4
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.chain_exec import TestScopes, setup_chain, exec_chain
from langchain_chroma import Chroma
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry, start_scope, stop_scope, monocle_trace_scope_method, monocle_trace_scope
from monocle_apptrace.instrumentation.common.utils import get_scopes
from monocle_apptrace.instrumentation.common.constants import CONFIG_FILE_NAME, CONFIG_FILE_PATH

CHAT_SCOPE_NAME = "chat"
custom_exporter = CustomConsoleSpanExporter()
@pytest.fixture(scope="module")
def setup():
    os.environ[CONFIG_FILE_PATH] = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE_NAME)
    setup_monocle_telemetry(
                workflow_name="langchain_app_1",
                span_processors=[SimpleSpanProcessor(custom_exporter)],
                wrapper_methods=[])

@pytest.fixture(autouse=True)
def pre_test():
    # clear old spans
    custom_exporter.reset()

@pytest.mark.integration()
def test_scope_api(setup):
    """ Test setting scope via start/stop API. Verify that the scope is effective across chains/traces, and not in effect after stop is called"""
    rag_chain = setup_chain()
    scope_name = "message"
    token = start_scope(scope_name)

    # 1st chain run
    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)
    message_scope_id = None
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+scope_name)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+scope_name)

    # 2nd chain run
    custom_exporter.reset() ## clear old spans
    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        assert span_attributes.get("scope."+scope_name) == message_scope_id

    stop_scope(token)
    
    # 3rd chain run
    custom_exporter.reset() ## clear old spans
    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        assert span_attributes.get("scope."+scope_name) is None

@pytest.mark.integration()
def test_scope_api_with_value(setup):
    """ Test setting scope via start/stop API with specific scope value """
    rag_chain = setup_chain()
    scope_name = "dummy"
    scope_value = "test123"
    token = start_scope(scope_name, scope_value)

    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)
    message_scope_id = None
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+scope_name)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+scope_name)
        assert message_scope_id == scope_value
    stop_scope(token)

@monocle_trace_scope_method(scope_name=CHAT_SCOPE_NAME)
def run_chain_with_scope(chain, message):
    result = chain.invoke(message)
    print(result)
    return result

@pytest.mark.integration()
def test_scope_wrapper(setup):
    """ Test setting scope at function level using decorator """
    rag_chain = setup_chain()
    scope_name = CHAT_SCOPE_NAME
    result = run_chain_with_scope(rag_chain, "What is Task Decomposition?")
    print(result)
    spans = custom_exporter.get_captured_spans()
    message_scope_id = None
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+scope_name)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+scope_name)

@pytest.mark.integration()
def test_scope_config(setup):
    """ Test setting scope at function level using external configuartion """
    test_scope = TestScopes()
    # set config path as monocle_scopes.json in the same directory as this file
    chain = setup_chain()
    chain = setup_chain()
    result = test_scope.config_scope_func(chain, "What is Task Decomposition?")
    print(result)
    scope_name = "question"
    spans = custom_exporter.get_captured_spans()
    message_scope_id = None
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+scope_name)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+scope_name)

@pytest.mark.integration()
def test_scope_with_code_block(setup):
    """ Test setting scope with code block """
    CODE_SCOPE_NAME = "chitchat"
    with monocle_trace_scope(CODE_SCOPE_NAME):
        response = exec_chain("What is Task Decomposition?")
        print(response)
    spans = custom_exporter.get_captured_spans()
    message_scope_id = None
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+CODE_SCOPE_NAME)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+CODE_SCOPE_NAME)

