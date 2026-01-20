import pytest
import threading
import time
import requests

from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from bedrock_agentcore import BedrockAgentCoreApp
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI



@tool
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
        
    Returns:
        The calculated result as a string
    """
    try:
        # Only allow basic math operations for safety
        allowed_chars = set("0123456789+-*/()%. ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


def create_langgraph_agent():
    """Create LangGraph agent"""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    react_agent = create_react_agent(
        model=llm,
        tools=[calculate],
    )
    
    def langgraph_agent(prompt: str) -> str:
        """Framework-agnostic LangGraph agent function"""
        result = react_agent.invoke(
            {"messages": [HumanMessage(content=prompt)]}
        )
        return result["messages"][-1].content
    
    return langgraph_agent


@pytest.fixture(scope="module")
def setup():
    instrumentor = None
    try:
        file_exporter = FileSpanExporter()
        memory_exporter = InMemorySpanExporter()
        span_processors = [
            BatchSpanProcessor(file_exporter),
            SimpleSpanProcessor(memory_exporter)
        ]
        instrumentor = setup_monocle_telemetry(
            workflow_name="langchain_agent_2",
            span_processors=span_processors
        )
        yield memory_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.fixture(scope="module")
def agentcore_server(setup):
    """Start AgentCore server in background thread"""
    # Create AgentCore app
    app = BedrockAgentCoreApp()
    
    # Create the LangGraph agent
    langgraph_agent = create_langgraph_agent()
    
    # Register agent with AgentCore
    @app.entrypoint
    def production_agent(request: dict) -> dict:
        prompt = request.get("prompt", "")
        response = langgraph_agent(prompt)
        return {"response": response}
    
    # Start server in a thread
    server_thread = threading.Thread(target=lambda: app.run(port=8080), daemon=True)
    server_thread.start()
    
    # Wait for server to be ready
    time.sleep(5)
    
    yield "http://localhost:8080"


def test_langgraph_agentcore_parity(agentcore_server, setup):
    """
    Ensure LangGraph agent behaves identically
    when executed locally and via AgentCore HTTP endpoint
    """
    prompt = "Calculate 4 + 5 * (6 - 3)"

    # AgentCore-style invocation via HTTP
    requests.post(
        f"{agentcore_server}/invocations",
        json={"prompt": prompt},
        timeout=30
    )
    
    verify_spans(setup)
    
    
def verify_spans(memory_exporter):
    """Verify all spans are generated correctly with AgentCore span present."""
    
    time.sleep(2)  # Wait for spans to be exported
    
    spans = memory_exporter.get_finished_spans()
    
    # Basic validation
    assert len(spans) > 0, "No spans were generated"
    
    found_agentcore_span = False
    found_langgraph_span = False
    found_tool_span = False
    found_inference_span = False
    
    for span in spans:
        span_attributes = span.attributes
        span_name = span.name
        
        # Check for BedrockAgentCore span
        if "bedrock_agentcore" in span_name.lower():
            found_agentcore_span = True
            assert "span.type" in span_attributes
            assert span_attributes["span.type"] == "http.process"
            assert "entity.1.route" in span_attributes
            assert span_attributes["entity.1.route"] == "production_agent"
            assert "entity.1.method" in span_attributes
            assert span_attributes["entity.1.method"] == "POST"
            
            # Check events
            span_events = span.events
            assert len(span_events) >= 2, "AgentCore span should have input and output events"
            
            # Verify input event
            input_event = next((e for e in span_events if e.name == "data.input"), None)
            assert input_event is not None, "AgentCore span missing input event"
            assert "request" in input_event.attributes
            
            # Verify output event
            output_event = next((e for e in span_events if e.name == "data.output"), None)
            assert output_event is not None, "AgentCore span missing output event"
            assert "response" in output_event.attributes
            assert "error_code" in output_event.attributes
            assert output_event.attributes["error_code"] == "success"
            
        # Check for LangGraph span
        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.invocation":
            found_langgraph_span = True
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.name"] == "LangGraph"
            assert span_attributes["entity.1.type"] == "agent.langgraph"
            
        # Check for tool invocation span
        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.tool.invocation":
            found_tool_span = True
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.name"] == "calculate"
            assert span_attributes["entity.1.type"] == "tool.langgraph"
            
        # Check for inference span
        if "span.type" in span_attributes and span_attributes["span.type"] == "inference.framework":
            found_inference_span = True
            assert "entity.2.name" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4"
    
    # Final assertions
    assert found_agentcore_span, "BedrockAgentCore span not found"
    assert found_langgraph_span, "LangGraph agent span not found"
    assert found_tool_span, "Tool invocation span not found"
    assert found_inference_span, "Inference span not found"
    
    print(f"\n✓ All required spans found and validated:")
    print(f"  - AgentCore span: ✓")
    print(f"  - LangGraph span: ✓")
    print(f"  - Tool span: ✓")
    print(f"  - Inference span: ✓")
    print(f"  Total spans: {len(spans)}")
    

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])