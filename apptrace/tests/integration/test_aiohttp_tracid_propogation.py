import logging
import os
import uuid

import aiohttp
import pytest
import pytest_asyncio
from common import aiohttp_helper
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.constants import (
    SCOPE_CONFIG_PATH,
    SCOPE_METHOD_FILE,
    TRACE_PROPOGATION_URLS,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.span_handler import http_span_counter

logger = logging.getLogger(__name__)

@pytest_asyncio.fixture
async def aiohttp_server(scope="function", autouse=False):
    logger.info("Setting up aiohttp server")
    os.environ[TRACE_PROPOGATION_URLS] = "http://localhost:8081"
    os.environ[SCOPE_CONFIG_PATH] = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCOPE_METHOD_FILE)
    custom_exporter = CustomConsoleSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="aiohttp_test",
            span_processors=[SimpleSpanProcessor(custom_exporter)]
        )
        
        runner = await aiohttp_helper.run_server()

        yield custom_exporter
        await runner.cleanup()
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

# per test fixture to clear spans
@pytest_asyncio.fixture
async def clear_spans(aiohttp_server):
    aiohttp_server.reset()
    http_span_counter.reset()
    yield



@pytest.mark.asyncio
async def test_chat_endpoint(clear_spans, aiohttp_server):
    url = aiohttp_helper.get_url()
    timeout = aiohttp.ClientTimeout(total=60)
    headers = {"client-id": str(uuid.uuid4())}
    question = "What is Task Decomposition?"

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{url}/chat?question={question}", headers=headers) as resp:
            logger.info(f"Response status: {resp.status}")
            text = await resp.text()
            logger.info(f"Response text: {text}")
            assert resp.status == 200

    verify_scopes(aiohttp_server)

@pytest.mark.asyncio
async def test_verify_skip_health_check(clear_spans, aiohttp_server):
    url = aiohttp_helper.get_url()
    timeout = aiohttp.ClientTimeout(total=60)
    # Send three health check requests and verify that only first span is captured
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(3):
            async with session.get(f"{url}") as resp:
                assert resp.status == 200
    
    spans = aiohttp_server.get_captured_spans()
    assert len(spans) > 0, f"No health check spans were captured"
    health_check_spans = [span for span in spans if span.attributes.get("span.type") == "http.process"]
    assert len(health_check_spans) == 1, f"Expected only 1 health check span, but found {len(health_check_spans)}"

def verify_scopes(aiohttp_server):
    scope_name = "conversation"
    spans = aiohttp_server.get_captured_spans()
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
            assert span_output.attributes['error_code'] == "200"
        if span_attributes.get("span.type", "") == "http.process":
            span_input, span_output = span.events
            assert span_attributes.get("entity.1.method").lower() == "get"
            assert span_attributes.get("entity.1.route") is not None
            assert span_attributes.get("entity.1.url") is not None
            assert span_output.attributes['error_code'] == "200"
        if trace_id is None:
            trace_id = span.context.trace_id
        else:
            assert trace_id == span.context.trace_id
            
if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
