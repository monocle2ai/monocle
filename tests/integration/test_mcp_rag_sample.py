"""
Test MCP servers as RAG using OpenAI SDK with native MCP tools support.
This test demonstrates how to use OpenAI's native MCP integration with monocle tracing.
"""

import os
import logging
from openai import OpenAI
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from tests.common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = logging.getLogger(__name__)

# Set up memory exporter and file export
memory_exporter = CustomConsoleSpanExporter()
os.environ["MONOCLE_EXPORTER"] = "file"

def setup():
    """Setup monocle telemetry for tracing."""
    memory_exporter.reset()
    setup_monocle_telemetry(
        workflow_name="mcp_rag_test",
        span_processors=[
            # SimpleSpanProcessor(memory_exporter),
            # File export is handled via environment variable
        ],
    )

def test_mcp_monocle_search():
    """Test MCP with monocle search functionality."""
    
    client = OpenAI()
    
    query = "What is the extract_messages function used for in the monocle codebase?"
    
    print("=" * 60)
    print("MONOCLE MCP SEARCH TEST")
    print("=" * 60)
    print(f"Query: {query}")
    print("Calling OpenAI with MCP tools...")
    
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "monocle",
                    "server_url": "http://localhost:8000/search_text/mcp/",
                    "require_approval": "never",
                },
            ],
            input=query,
        )
        
        print(f"\nResponse:\n{resp.output_text}")
        print("=" * 60)
        return resp.output_text
        
    except Exception as e:
        logger.error(f"Error with MCP monocle search: {e}")
        print(f"Error: {e}")
        return f"Error: {e}"

def test_mcp_okahu_search():
    """Test MCP with Okahu search functionality."""
    
    client = OpenAI()
    
    query = "What apps are available that contain 'chatbot' in their name? Give me details about their configuration."
    
    print("=" * 60)
    print("OKAHU MCP SEARCH TEST")
    print("=" * 60)
    print(f"Query: {query}")
    print("Calling OpenAI with MCP tools...")
    
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "okahu",
                    "server_url": "http://localhost:8000/okahu/mcp/",
                    "require_approval": "never",
                },
            ],
            input=query,
        )
        
        print(f"\nResponse:\n{resp.output_text}")
        print("=" * 60)
        return resp.output_text
        
    except Exception as e:
        logger.error(f"Error with MCP okahu search: {e}")
        print(f"Error: {e}")
        return f"Error: {e}"

def test_mcp_multi_server():
    """Test MCP with multiple servers for comprehensive search."""
    
    client = OpenAI()
    
    query = "Search for any security-related functions or vulnerabilities in the monocle codebase. Also check if there are any security-related apps in Okahu."
    
    print("=" * 60)
    print("MULTI-SERVER MCP TEST")
    print("=" * 60)
    print(f"Query: {query}")
    print("Calling OpenAI with multiple MCP servers...")
    
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "monocle",
                    "server_url": "http://localhost:8000/search_text/mcp/",
                    "require_approval": "never",
                },
                {
                    "type": "mcp",
                    "server_label": "okahu",
                    "server_url": "http://localhost:8000/okahu/mcp/",
                    "require_approval": "never",
                },
            ],
            input=query,
        )
        
        print(f"\nResponse:\n{resp.output_text}")
        print("=" * 60)
        return resp.output_text
        
    except Exception as e:
        logger.error(f"Error with multi-server MCP search: {e}")
        print(f"Error: {e}")
        return f"Error: {e}"

def test_simple_mcp_query():
    """Test a simple MCP query to verify basic functionality."""
    
    client = OpenAI()
    
    query = "Search for 'FileSpanExporter' in the monocle codebase and explain what it does."
    
    print("=" * 60)
    print("SIMPLE MCP TEST")
    print("=" * 60)
    print(f"Query: {query}")
    
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "monocle",
                    "server_url": "http://localhost:8000/search_text/mcp/",
                    "require_approval": "never",
                },
            ],
            input=query,
        )
        
        print(f"\nResponse:\n{resp.output_text}")
        print("=" * 60)
        return resp.output_text
        
    except Exception as e:
        logger.error(f"Error with simple MCP query: {e}")
        print(f"Error: {e}")
        return f"Error: {e}"

def run_all_tests():
    """Run all MCP tests."""
    
    print("🚀 Starting MCP RAG Test Suite with OpenAI Native MCP Support...")
    print("This test demonstrates RAG using OpenAI's native MCP tools with monocle tracing\n")
    
    # Test 1: Simple MCP query
    print("🔍 Test 1: Simple MCP Query...")
    test_simple_mcp_query()
    print()
    
    # Test 2: Monocle search
    print("🔍 Test 2: Monocle Search...")
    test_mcp_monocle_search()
    print()
    
    # Test 3: Okahu search
    print("🔍 Test 3: Okahu Search...")
    test_mcp_okahu_search()
    print()
    
    # Test 4: Multi-server search
    print("🔍 Test 4: Multi-Server Search...")
    test_mcp_multi_server()
    print()

if __name__ == "__main__":
    # Setup monocle telemetry
    setup()
    
    # Run all tests
    run_all_tests()
    
    print("✅ All tests completed!")
    print("Check the generated trace files for telemetry data.")
