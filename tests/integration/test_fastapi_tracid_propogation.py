from common.custom_exporter import CustomConsoleSpanExporter
from common.chain_exec import TestScopes, setup_chain
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry, start_scope, stop_scope
from monocle_apptrace.instrumentation.common.constants import SCOPE_METHOD_FILE, SCOPE_CONFIG_PATH, TRACE_PROPOGATION_URLS
custom_exporter = CustomConsoleSpanExporter()
import pytest , pytest_asyncio
import aiohttp
from tests.common import fastapi_helper
import uuid
import os

# Check if FastAPI is available
try:
    from tests.common.fastapi_helper import FASTAPI_AVAILABLE
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI and uvicorn not available")

@pytest_asyncio.fixture
async def fastapi_server(scope="module"):
    print("Setting up FastAPI server")
    os.environ[TRACE_PROPOGATION_URLS] = "http://localhost:8083"
    os.environ[SCOPE_CONFIG_PATH] = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCOPE_METHOD_FILE)

    setup_monocle_telemetry(
        workflow_name="fastapi_test",
        span_processors=[SimpleSpanProcessor(custom_exporter)]
    )

    runner = await fastapi_helper.run_server()

    yield
    await runner.cleanup()

@pytest_asyncio.fixture(autouse=True)
def pre_test():
    # clear old spans
    custom_exporter.reset()

@pytest.mark.asyncio
async def test_chat_endpoint(fastapi_server):
    custom_exporter.reset()

    url = fastapi_helper.get_url()
    timeout = aiohttp.ClientTimeout(total=60)
    headers = {"client-id": str(uuid.uuid4())}
    question = "What is Task Decomposition?"

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{url}/chat?question={question}", headers=headers) as resp:
            print(f"Response status: {resp.status}")
            text = await resp.text()
            print(f"Response text: {text}")
            assert resp.status == 200

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
            span_input, span_output = span.events
            assert span_attributes.get("entity.1.method").lower() == "get"
            assert span_attributes.get("entity.1.route") is not None
            assert span_output.attributes['status'] == "200"
        if trace_id is None:
            trace_id = span.context.trace_id
        else:
            assert trace_id == span.context.trace_id
            
if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
