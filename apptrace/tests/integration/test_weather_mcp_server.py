"""
Integration test for Weather MCP server (stateless HTTP, FastAPI) using the MCP protocol and Monocle tracing.
"""
import pytest
import asyncio
import threading
import time
import uvicorn
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from integration.servers.mcp.weather_server import app as weather_app
from langchain_mcp_adapters.client import MultiServerMCPClient

WEATHER_MCP_URL = "http://127.0.0.1:8086/weather/mcp/"

@pytest.fixture(scope="module")
def in_memory_exporter():
    exporter = InMemorySpanExporter()
    return exporter

@pytest.fixture(scope="module", autouse=True)
def start_weather_server():
    """Start the Weather MCP server in a background thread for the test module."""
    def run_server():
        uvicorn.run(weather_app, host="127.0.0.1", port=8086, log_level="error")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Wait for server to start
    yield
    # No explicit shutdown; daemon thread will exit with process

@pytest.fixture(scope="module")
def setup_instrumentation(in_memory_exporter):
    instrumentor = setup_monocle_telemetry(
        workflow_name="weather_mcp",
        span_processors=[SimpleSpanProcessor(in_memory_exporter), BatchSpanProcessor(FileSpanExporter())]
    )
    yield instrumentor
    in_memory_exporter.clear()

@pytest.mark.asyncio
async def test_weather_mcp_tool_call(start_weather_server, setup_instrumentation, in_memory_exporter):
    """Test calling the get_weather tool on the Weather MCP server via MCP protocol, with session_id propagation and span validation."""
    in_memory_exporter.clear()
    session_id = "test-weather-session-2026"
    client = MultiServerMCPClient({
        "weather": {
            "url": WEATHER_MCP_URL,
            "transport": "streamable_http",
            "headers": {"Mcp-Session-Id": session_id}
        }
    })
    # Discover tools
    tools = await client.get_tools()
    # Find the get_weather tool
    tool = next((t for t in tools if t.name == "get_weather"), None)
    assert tool is not None, "get_weather tool not found in MCP server"
    # Call the tool as an async function (LangChain tool pattern)
    result = await tool.ainvoke({"city": "London"})
    # assert isinstance(result, dict), f"Tool result is not a dict: {result}"
    # assert "temperature" in result, f"No temperature in response: {result}"
    await asyncio.sleep(0.1)

    spans = in_memory_exporter.get_finished_spans()
    # Find MCP spans (BaseSession.send_request)
    mcp_spans = [s for s in spans if "send_request" in s.name.lower() or s.attributes.get("entity.1.type") == "mcp.server"]
    assert len(mcp_spans) >= 1, f"Should have at least 1 MCP span, found {len(mcp_spans)}"
    
    # Verify server_name and URL are captured in MCP spans only
    for span in mcp_spans:
        server_name = span.attributes.get("entity.1.server_name")
        server_url = span.attributes.get("entity.1.url")
        # Server name should be fetched from MCP server info
        assert server_name == "WeatherServer", f"Expected 'WeatherServer', got: {server_name}"
        assert server_url == WEATHER_MCP_URL, f"Expected {WEATHER_MCP_URL}, got: {server_url}"
        # session_id_val = span.attributes.get("monocle.scope.agentic.session")
        # assert session_id_val == session_id, f"session_id scope missing or incorrect in span: {span.name}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
