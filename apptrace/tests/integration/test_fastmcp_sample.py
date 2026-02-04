"""
Integration test for FastMCP instrumentation with actual server and client.
Tests that spans are generated when calling MCP tools.
"""
import pytest
import asyncio
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.exporters.file_exporter import FileSpanExporter


@pytest.fixture(scope="module")
def in_memory_exporter():
    """Create an in-memory span exporter to capture spans"""
    exporter = InMemorySpanExporter()
    return exporter


@pytest.fixture(scope="module")
def setup_instrumentation(in_memory_exporter):
    """Setup Monocle instrumentation with in-memory exporter"""
    
    # Setup Monocle telemetry
    instrumentor = setup_monocle_telemetry(
        workflow_name="fastmcp",
        span_processors=[SimpleSpanProcessor(in_memory_exporter), BatchSpanProcessor(FileSpanExporter())]
    )
    
    yield instrumentor
    
    # Cleanup
    in_memory_exporter.clear()


@pytest.fixture
def mcp_server(setup_instrumentation):
    """Create and return a FastMCP server instance"""
    # Import FastMCP after instrumentation is setup
    # This ensures the instrumentation wraps the methods correctly
    from fastmcp import FastMCP
    
    mcp = FastMCP("Test Server", host="127.0.0.1", port=8082)
    
    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b
    
    @mcp.tool
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers"""
        return x * y
    
    @mcp.resource("file://test/data.txt")
    def get_test_data() -> str:
        """Get test data"""
        return "This is test data from a resource"
    
    @mcp.resource("file://test/config.json")
    def get_config() -> str:
        """Get configuration"""
        return '{"setting": "value", "enabled": true}'
    
    @mcp.prompt()
    def greeting_prompt(name: str = "User") -> str:
        """Generate a greeting prompt"""
        return f"Say hello to {name} in a friendly way"
    
    @mcp.prompt()
    def analysis_prompt(topic: str) -> str:
        """Generate an analysis prompt"""
        return f"Provide a detailed analysis of {topic}"
        
    return mcp


@pytest.mark.asyncio
async def test_all_fastmcp_operations(mcp_server, in_memory_exporter):
    """Test that all MCP operations (tool, resource, prompt) generate correct spans"""
    # Clear existing spans
    in_memory_exporter.clear()
    
    # Perform all types of operations
    try:
        tool_result = await mcp_server._call_tool_mcp("add", {"a": 5, "b": 3})
        print(f"\n Tool result: {tool_result}")
    except Exception as e:
        print(f"\n  Tool error: {e}")
    
    try:
        resource_result = await mcp_server._read_resource_mcp("file://test/config.json")
        print(f"\n Resource result: {resource_result}")
    except Exception as e:
        print(f"\n  Resource error: {e}")
    
    try:
        prompt_result = await mcp_server._get_prompt_mcp("analysis_prompt", {"topic": "AI"})
        print(f"\n Prompt result: {prompt_result}")
    except Exception as e:
        print(f"\n  Prompt error: {e}")
    
    # Give a moment for spans to be processed
    await asyncio.sleep(0.1)
    
    # Get the captured spans
    spans = in_memory_exporter.get_finished_spans()
    
    # Verify we have spans for each operation type
    tool_spans = [s for s in spans if "tool" in s.name.lower()]
    resource_spans = [s for s in spans if "resource" in s.name.lower()]
    prompt_spans = [s for s in spans if "prompt" in s.name.lower()]

    
    assert len(tool_spans) >= 1, "Should have at least 1 tool span"
    assert len(resource_spans) >= 1, "Should have at least 1 resource span"
    assert len(prompt_spans) >= 1, "Should have at least 1 prompt span"
    
    # Assert server name and server url in tool, resource, and prompt spans
    for span in spans:
        if any(key in span.name.lower() for key in ("tool", "resource", "prompt")):
            server_name = span.attributes.get("entity.1.server_name")
            server_url = span.attributes.get("entity.1.url")
            assert server_name == "Test Server", f"Incorrect server name in span: {span.name}"
            assert server_url == "http://127.0.0.1:8082", f"Incorrect server url in span: {span.name}"    

@pytest.mark.asyncio
async def test_all_list_operations_together(mcp_server, in_memory_exporter):
    """Test that all list operations generate correct spans"""
    # Clear existing spans
    in_memory_exporter.clear()
    
    # Perform all list operations
    try:
        tools_result = await mcp_server._list_tools_mcp()
        print(f"\n Tools: {len(tools_result.tools) if hasattr(tools_result, 'tools') else 'N/A'}")
    except Exception as e:
        print(f"\n  List tools error: {e}")
    
    try:
        resources_result = await mcp_server._list_resources_mcp()
        print(f"\n Resources: {len(resources_result.resources) if hasattr(resources_result, 'resources') else 'N/A'}")
    except Exception as e:
        print(f"\n  List resources error: {e}")
    
    try:
        prompts_result = await mcp_server._list_prompts_mcp()
        print(f"\n Prompts: {len(prompts_result.prompts) if hasattr(prompts_result, 'prompts') else 'N/A'}")
    except Exception as e:
        print(f"\n  List prompts error: {e}")
    
    # Give a moment for spans to be processed
    await asyncio.sleep(0.1)
    
    # Get the captured spans
    spans = in_memory_exporter.get_finished_spans()
    
    # Verify we have spans for each list operation type
    tools_list_spans = [s for s in spans if "tools_list" in s.name.lower()]
    resources_list_spans = [s for s in spans if "resources_list" in s.name.lower()]
    prompts_list_spans = [s for s in spans if "prompts_list" in s.name.lower()]
    
    assert len(tools_list_spans) >= 1, "Should have at least 1 tools list span"
    assert len(resources_list_spans) >= 1, "Should have at least 1 resources list span"
    assert len(prompts_list_spans) >= 1, "Should have at least 1 prompts list span"
    
    print(f"\n All list operations successfully instrumented!")
    
    for span in spans:
        if any(key in span.name.lower() for key in ("tool", "resource", "prompt")):
            server_name = span.attributes.get("entity.1.server_name")
            server_url = span.attributes.get("entity.1.url")
            assert server_name == "Test Server", f"Incorrect server name in span: {span.name}"
            assert server_url == "http://127.0.0.1:8082", f"Incorrect server url in span: {span.name}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
