import os
import bs4
import pytest
import asyncio
from common.custom_exporter import CustomConsoleSpanExporter
from common.chain_exec import TestScopes, setup_chain, exec_chain
from langchain_chroma import Chroma
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace import setup_monocle_telemetry, start_scope, stop_scope, monocle_trace_scope_method, monocle_trace_scope
from monocle_apptrace.instrumentation.common.utils import get_scopes
from monocle_apptrace.instrumentation.common.constants import SCOPE_METHOD_FILE, SCOPE_CONFIG_PATH

CHAT_SCOPE_NAME = "chat"
custom_exporter = CustomConsoleSpanExporter()
@pytest.fixture(scope="module")
def setup():
    os.environ[SCOPE_CONFIG_PATH] = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCOPE_METHOD_FILE)
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
    scope_name = "message"
    token = start_scope(scope_name)
    rag_chain = setup_chain()
 
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
        print(span_attributes)
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
    scope_name = "dummy"
    scope_value = "test123"
    token = start_scope(scope_name, scope_value)
    rag_chain = setup_chain()

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
def run_chain_with_scope(message):
    chain = setup_chain()    
    result = chain.invoke(message)
    print(result)
    return result

@pytest.mark.integration()
def test_scope_wrapper(setup):
    """ Test setting scope at function level using decorator """
    result = run_chain_with_scope("What is Task Decomposition?")
    verify_scope_testing(scope_name = CHAT_SCOPE_NAME)

@monocle_trace_scope_method(scope_name=CHAT_SCOPE_NAME)
async def run_chain_async_with_scope(message):
    chain = setup_chain()
    result = chain.invoke(message)
    print(result)
    return result

@pytest.mark.integration()
def test_async_scope_wrapper(setup):
    """ Test setting scope at async function level using decorator """
    result = asyncio.run(run_chain_async_with_scope("What is Task Decomposition?"))
    verify_scope_testing(scope_name = CHAT_SCOPE_NAME)

def verify_scope_testing(scope_name:str):
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
    result = test_scope.config_scope_func("What is Task Decomposition?")
    verify_scope_testing(scope_name = "question")

@pytest.mark.integration()
def test_async_scope_config(setup):
    """ Test setting scope at function level using external configuartion """
    test_scope = TestScopes()
    # set config path as monocle_scopes.json in the same directory as this file
    result = asyncio.run(test_scope.config_scope_async_func("What is Task Decomposition?"))
    verify_scope_testing(scope_name = "aquestion")

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

