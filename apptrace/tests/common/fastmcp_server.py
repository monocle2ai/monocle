from fastmcp import FastMCP
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

# Setup Monocle instrumentation
setup_monocle_telemetry(
    workflow_name="fastmcp_test_server",
    monocle_exporters_list='file'
)

mcp = FastMCP("Demo")

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

if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8000)