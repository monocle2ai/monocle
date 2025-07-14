"""
Test MCP servers as RAG using OpenAI Agents SDK with native MCP support.
This test demonstrates how to use OpenAI Agents SDK with MCP integration and monocle tracing.
"""

import os
import logging
import asyncio
import pytest
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner
from agents.mcp.server import MCPServerStreamableHttp
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from tests.common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = logging.getLogger(__name__)

# Set up memory exporter and file export
memory_exporter = CustomConsoleSpanExporter()
os.environ["MONOCLE_EXPORTER"] = "file, okahu"

@pytest.fixture(scope="session", autouse=True)
def setup_monocle():
    """Setup monocle telemetry for tracing."""
    memory_exporter.reset()
    setup_monocle_telemetry(
        workflow_name="mcp_agents_test",
        span_processors=[
            # SimpleSpanProcessor(memory_exporter),
            # File export is handled via environment variable
        ],
    )

@pytest.mark.asyncio
async def test_mcp_monocle_agent():
    """Test OpenAI Agent with monocle MCP server."""
    
    print("=" * 60)
    print("MONOCLE MCP AGENT TEST")
    print("=" * 60)
    
    try:
        # Create MCP server for monocle
        monocle_server = MCPServerStreamableHttp(
            params={
                "url": "http://localhost:8000/search_text/mcp/",                
            },
            cache_tools_list=True
            # url="http://localhost:8000/search_text/mcp/",
        )

        await monocle_server.connect()

        # Create agent with MCP server
        agent = Agent(
            name="Monocle Assistant",
            instructions="You are a helpful assistant that can search through the monocle codebase. Use the available tools to find information about functions, classes, and code patterns.",
            model="gpt-4o-mini",
            mcp_servers=[monocle_server]
        )
        
        query = "Search the monocle codebase for the 'extract_messages' function and explain what it does."
        print(f"Query: {query}")
        print("Processing with OpenAI Agent...")
        
        # Run the agent
        response = Runner.run_sync(agent, query)

        print(f"\nAgent Response:\n{response.final_output}")
        print("=" * 60)

        return response.final_output

    except Exception as e:
        logger.error(f"Error with monocle MCP agent: {e}")
        print(f"Error: {e}")
        return f"Error: {e}"

@pytest.mark.asyncio
async def test_mcp_okahu_agent():
    """Test OpenAI Agent with Okahu MCP server."""
    
    print("=" * 60)
    print("OKAHU MCP AGENT TEST")
    print("=" * 60)
    
    try:
        # Create MCP server for okahu
        okahu_server = MCPServerStreamableHttp(
            params={
                "url": "http://localhost:8000/okahu/mcp/",
            },
            cache_tools_list=True
        )

        await okahu_server.connect()

        # Create agent with MCP server
        agent = Agent(
            name="Okahu Assistant",
            instructions="You are a helpful assistant that can search through Okahu applications and services. Use the available tools to find information about apps, configurations, and monitoring data.",
            model="gpt-4o-mini",
            mcp_servers=[okahu_server]
        )
        
        query = "Find apps that contain 'chatbot' in their name and provide details about their configuration."
        print(f"Query: {query}")
        print("Processing with OpenAI Agent...")
        
        # Run the agent
        response = await Runner.run(agent, query)

        print(f"\nAgent Response:\n{response.final_output}")
        print("=" * 60)
        
        return response.final_output
        
    except Exception as e:
        logger.error(f"Error with Okahu MCP agent: {e}")
        print(f"Error: {e}")
        return f"Error: {e}"

@pytest.mark.asyncio
async def test_mcp_multi_server_agent():
    """Test OpenAI Agent with multiple MCP servers."""
    
    print("=" * 60)
    print("MULTI-SERVER MCP AGENT TEST")
    print("=" * 60)
    
    try:
        # Create MCP servers
        monocle_server = MCPServerStreamableHttp(
            params={
                "url": "http://localhost:8000/search_text/mcp/",                
            },
            cache_tools_list=True
        )
        
        okahu_server = MCPServerStreamableHttp(
            params={
                "url": "http://localhost:8000/okahu/mcp/",
            },
            cache_tools_list=True
        )
        
        await monocle_server.connect()
        await okahu_server.connect()
        
        # Create agent with multiple MCP servers
        agent = Agent(
            name="Multi-Server Assistant",
            instructions="You are a helpful assistant with access to both monocle codebase search and Okahu applications. Use the appropriate tools to provide comprehensive answers.",
            model="gpt-4o-mini",
            mcp_servers=[monocle_server, okahu_server]
        )
        
        query = """Perform a comprehensive security analysis:
        1. Search the monocle codebase for any security-related functions or potential vulnerabilities
        2. Check Okahu for any security-related applications or monitoring tools
        3. Provide a summary of security features and recommendations"""
        
        print(f"Query: {query}")
        print("Processing with multi-server OpenAI Agent...")
        
        # Run the agent
        response = await Runner.run(agent, query)

        print(f"\nAgent Response:\n{response.final_output}")
        print("=" * 60)
        
        return response.final_output
        
    except Exception as e:
        logger.error(f"Error with multi-server MCP agent: {e}")
        print(f"Error: {e}")
        return f"Error: {e}"

@pytest.mark.asyncio
async def test_mcp_streaming_agent():
    """Test OpenAI Agent with streaming response."""
    
    print("=" * 60)
    print("STREAMING MCP AGENT TEST")
    print("=" * 60)
    
    try:
        # Create MCP server
        monocle_server = MCPServerStreamableHttp(
            params={
                "url": "http://localhost:8000/search_text/mcp/",
            },
            cache_tools_list=True
        )

        await monocle_server.connect()

        # Create agent with MCP server
        agent = Agent(
            name="Streaming Assistant",
            instructions="You are a helpful assistant that can search through the monocle codebase. Provide detailed explanations about code functionality.",
            model="gpt-4o-mini",
            mcp_servers=[monocle_server]
        )
        
        query = "Search for FileSpanExporter in the monocle codebase and explain its purpose and functionality."
        print(f"Query: {query}")
        print("Streaming response from OpenAI Agent...")
        print("\nStreaming Response:")
        
        # Stream the response
        full_response = ""
        async for event in Runner.run_streamed(agent, query).stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
        
        print("\n" + "=" * 60)
        
        return full_response
        
    except Exception as e:
        logger.error(f"Error with streaming MCP agent: {e}")
        print(f"Error: {e}")
        return f"Error: {e}"

@pytest.mark.asyncio
async def test_simple_mcp_agent():
    """Test a simple MCP agent to verify basic functionality."""
    
    print("=" * 60)
    print("SIMPLE MCP AGENT TEST")
    print("=" * 60)
    
    try:
        # Create MCP server
        monocle_server = MCPServerStreamableHttp(
            params={
                "url": "http://localhost:8000/search_text/mcp/",
            },
            cache_tools_list=True
        )

        await monocle_server.connect()

        # Create agent with MCP server
        agent = Agent(
            name="Simple Assistant",
            instructions="You are a helpful assistant. Use the search tools to find information in the monocle codebase.",
            model="gpt-4o-mini",
            mcp_servers=[monocle_server]
        )
        
        query = "Search for 'FileSpanExporter' in the monocle codebase and explain what it does."
        print(f"Query: {query}")
        print("Processing with OpenAI Agent...")
        
        # Run the agent
        response = await Runner.run(agent, query)

        print(f"\nAgent Response:\n{response.final_output}")
        print("=" * 60)
        
        return response.final_output
        
    except Exception as e:
        logger.error(f"Error with simple MCP agent: {e}")
        print(f"Error: {e}")
        return f"Error: {e}"

@pytest.mark.asyncio
async def test_run_all_agent_tests():
    """Run all OpenAI Agents MCP tests."""
    
    print("🤖 Starting OpenAI Agents MCP Test Suite...")
    print("This test demonstrates OpenAI Agents SDK with MCP integration and monocle tracing\n")
    
    # Test 1: Simple MCP agent
    print("🔍 Test 1: Simple MCP Agent...")
    await test_simple_mcp_agent()
    print()
    
    # Test 2: Basic monocle search
    print("🔍 Test 2: Monocle MCP Agent...")
    await test_mcp_monocle_agent()
    print()
    
    # Test 3: Okahu search
    print("🔍 Test 3: Okahu MCP Agent...")
    await test_mcp_okahu_agent()
    print()
    
    # Test 4: Multi-server agent
    print("🔍 Test 4: Multi-Server MCP Agent...")
    await test_mcp_multi_server_agent()
    print()
    
    # Test 5: Streaming response
    print("🔍 Test 5: Streaming MCP Agent...")
    await test_mcp_streaming_agent()
    print()

# if __name__ == "__main__":
#     # For running the tests directly with pytest
#     import pytest
#     import sys
    
#     print("✅ Use 'pytest' to run these async tests!")
#     print("Example: pytest tests/integration/test_mcp_agents_sample.py -v")
#     sys.exit(0)

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])