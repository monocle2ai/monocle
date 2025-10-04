import asyncio
import logging
import os

import pytest
from common.chain_exec import TestScopes, exec_chain, setup_chain
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace import (
    monocle_trace_scope,
    monocle_trace_scope_method,
    setup_monocle_telemetry,
    start_scope,
    stop_scope,
)
from monocle_apptrace.instrumentation.common.constants import (
    SCOPE_CONFIG_PATH,
    SCOPE_METHOD_FILE,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = logging.getLogger(__name__)
CHAT_SCOPE_NAME = "chat"

@pytest.fixture(scope="function")
def setup():
    # Save original environment variable value
    original_scope_config = os.environ.get(SCOPE_CONFIG_PATH)
    instrumentor = None
    try:
        custom_exporter = CustomConsoleSpanExporter()
        os.environ[SCOPE_CONFIG_PATH] = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCOPE_METHOD_FILE)
        instrumentor = setup_monocle_telemetry(
                    workflow_name="langchain_app_1",
                    span_processors=[SimpleSpanProcessor(custom_exporter)],
                    wrapper_methods=[])
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        # Restore original environment variable value or remove if it wasn't set
        if original_scope_config is not None:
            os.environ[SCOPE_CONFIG_PATH] = original_scope_config
        elif SCOPE_CONFIG_PATH in os.environ:
            del os.environ[SCOPE_CONFIG_PATH]



def test_scope_api(setup):
    """ Test setting scope via start/stop API. Verify that the scope is effective across chains/traces, and not in effect after stop is called"""
    scope_name = "message"
    token = start_scope(scope_name)
    rag_chain = setup_chain()
 
    # 1st chain run
    result = rag_chain.invoke("What is Task Decomposition?")
    logger.info(result)
    message_scope_id = None
    spans = setup.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+scope_name)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+scope_name)

    # 2nd chain run
    setup.reset() ## clear old spans
    result = rag_chain.invoke("What is Task Decomposition?")
    logger.info(result)
    spans = setup.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        logger.info(span_attributes)
        assert span_attributes.get("scope."+scope_name) == message_scope_id

    stop_scope(token)
    
    # 3rd chain run
    setup.reset() ## clear old spans
    result = rag_chain.invoke("What is Task Decomposition?")
    logger.info(result)
    spans = setup.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        assert span_attributes.get("scope."+scope_name) is None

def test_scope_api_with_value(setup):
    """ Test setting scope via start/stop API with specific scope value """
    scope_name = "dummy"
    scope_value = "test123"
    token = start_scope(scope_name, scope_value)
    rag_chain = setup_chain()

    result = rag_chain.invoke("What is Task Decomposition?")
    logger.info(result)
    message_scope_id = None
    spans = setup.get_captured_spans()
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
    logger.info(result)
    return result

def test_scope_wrapper(setup):
    """ Test setting scope at function level using decorator """
    result = run_chain_with_scope("What is Task Decomposition?")
    verify_scope_testing(setup, scope_name = CHAT_SCOPE_NAME)

@monocle_trace_scope_method(scope_name=CHAT_SCOPE_NAME)
async def run_chain_async_with_scope(message):
    chain = setup_chain()
    result = chain.invoke(message)
    logger.info(result)
    return result

def test_async_scope_wrapper(setup):
    """ Test setting scope at async function level using decorator """
    result = asyncio.run(run_chain_async_with_scope("What is Task Decomposition?"))
    verify_scope_testing(setup, scope_name = CHAT_SCOPE_NAME)

def verify_scope_testing(setup, scope_name:str):
    spans = setup.get_captured_spans()
    message_scope_id = None
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+scope_name)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+scope_name)

def test_scope_config(setup):
    """ Test setting scope at function level using external configuartion """
    test_scope = TestScopes()
    # set config path as monocle_scopes.json in the same directory as this file
    result = test_scope.config_scope_func("What is Task Decomposition?")
    verify_scope_testing(setup, scope_name = "question")

def test_async_scope_config(setup):
    """ Test setting scope at function level using external configuartion """
    test_scope = TestScopes()
    # set config path as monocle_scopes.json in the same directory as this file
    result = asyncio.run(test_scope.config_scope_async_func("What is Task Decomposition?"))
    verify_scope_testing(setup, scope_name = "aquestion")

def test_scope_with_code_block(setup):
    """ Test setting scope with code block """
    CODE_SCOPE_NAME = "chitchat"
    with monocle_trace_scope(CODE_SCOPE_NAME):
        response = exec_chain("What is Task Decomposition?")
        logger.info(response)
    spans = setup.get_captured_spans()
    message_scope_id = None
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+CODE_SCOPE_NAME)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+CODE_SCOPE_NAME)

