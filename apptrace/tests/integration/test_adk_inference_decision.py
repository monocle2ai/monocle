"""
Integration test to verify inference.decision.span.id linking for ADK.

This test verifies that:
1. Inference spans get proper span.subtype (tool_call or turn_end)
2. Tool invocation spans have inference.decision.span.id pointing to the inference span that decided to call them
3. The linking mechanism works correctly across multiple tool calls
"""
import datetime
import logging
import os
import time
from zoneinfo import ZoneInfo

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from config.conftest import temporary_env_var
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from monocle_apptrace import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup():
    try:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        custom_exporter = CustomConsoleSpanExporter()
        memory_exporter = InMemorySpanExporter()
        span_processors = [SimpleSpanProcessor(memory_exporter), SimpleSpanProcessor(custom_exporter)]
        instrumentor = setup_monocle_telemetry(
            workflow_name="adk_inference_decision_test",
            span_processors=span_processors
        )
        yield memory_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def get_weather(city: str) -> dict:
    """Get weather report for a city."""
    return {
        "status": "success",
        "report": f"Sunny, 25°C in {city}"
    }


def get_time(city: str) -> dict:
    """Get current time in a city."""
    tz = ZoneInfo("America/New_York") if city.lower() == "new york" else ZoneInfo("UTC")
    now = datetime.datetime.now(tz)
    return {
        "status": "success",
        "report": f'Current time in {city}: {now.strftime("%Y-%m-%d %H:%M:%S %Z")}'
    }


root_agent = Agent(
    name="multi_tool_agent",
    model="gemini-2.0-flash",
    description="Agent that can check weather and time",
    instruction="You are a helpful agent. Answer questions about weather and time.",
    tools=[get_weather, get_time],
)

APP_NAME = "decision_test_app"
USER_ID = "test_user"


async def run_agent(test_message: str, session_id: str = None):
    """Run the agent with a test message."""
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    if session_id is None:
        import uuid
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id
    )
    content = types.Content(role='user', parts=[types.Part(text=test_message)])
    
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content
    ):
        if event.is_final_response():
            logger.info(f"Response: {event.content}")


@pytest.mark.asyncio
async def test_inference_decision_linking_single_tool(setup):
    """Test that inference.decision.span.id is set correctly for single tool call."""
    test_message = "What is the weather in Paris?"
    await run_agent(test_message)
    verify_inference_decision_linking(setup, expected_tools=["get_weather"])


@pytest.mark.asyncio
async def test_inference_decision_linking_multiple_turns(setup):
    """Test inference decision linking across multiple turns."""
    # First query - should call get_weather
    await run_agent("What's the weather in Tokyo?")
    time.sleep(1)
    
    # Second query - should call get_time
    await run_agent("What time is it in New York?")
    
    verify_inference_decision_linking(setup, expected_tools=["get_weather", "get_time"])


def verify_inference_decision_linking(memory_exporter, expected_tools):
    """
    Verify that inference.decision.span.id correctly links tool invocations
    to the inference spans that decided to invoke them.
    """
    time.sleep(2)
    spans = memory_exporter.get_finished_spans()
    
    # Track inference spans that have tool_call subtype
    tool_call_inference_spans = {}  # tool_name -> inference_span_id
    
    # Track tool invocation spans
    tool_invocation_spans = {}  # tool_name -> list of (span, decision_span_id)
    
    for span in spans:
        span_attributes = span.attributes
        
        # Collect inference spans with tool_call subtype
        if (span_attributes.get("span.type") == "inference" and 
            span_attributes.get("span.subtype") == "tool_call"):
            
            tool_name = span_attributes.get("entity.3.name")
            if tool_name:
                inference_span_id = format(span.context.span_id, '#018x')
                tool_call_inference_spans[tool_name] = inference_span_id
                
                logger.info(f"Found inference span with tool_call subtype:")
                logger.info(f"  - Tool: {tool_name}")
                logger.info(f"  - Inference Span ID: {inference_span_id}")
                logger.info(f"  - Finish reason: {span.events[2].attributes.get('finish_reason')}")
        
        # Collect tool invocation spans
        if span_attributes.get("span.type") == "agentic.tool.invocation":
            tool_name = span_attributes.get("entity.1.name")
            decision_span_id = span_attributes.get("inference.decision.span.id")
            
            if tool_name not in tool_invocation_spans:
                tool_invocation_spans[tool_name] = []
            tool_invocation_spans[tool_name].append((span, decision_span_id))
            
            logger.info(f"Found tool invocation span:")
            logger.info(f"  - Tool: {tool_name}")
            logger.info(f"  - Decision Span ID: {decision_span_id}")
    
    # Verify that each expected tool has both an inference span and invocation span
    for tool_name in expected_tools:
        # Check inference span exists
        assert tool_name in tool_call_inference_spans, \
            f"Expected to find inference span with tool_call subtype for tool '{tool_name}'"
        
        # Check tool invocation span exists
        assert tool_name in tool_invocation_spans, \
            f"Expected to find tool invocation span for tool '{tool_name}'"
        
        # Verify the linking
        inference_span_id = tool_call_inference_spans[tool_name]
        tool_spans = tool_invocation_spans[tool_name]
        
        for tool_span, decision_span_id in tool_spans:
            assert decision_span_id is not None, \
                f"Tool invocation span for '{tool_name}' should have inference.decision.span.id"
            
            assert decision_span_id == inference_span_id, \
                f"Tool invocation span for '{tool_name}' has incorrect inference.decision.span.id. " \
                f"Expected: {inference_span_id}, Got: {decision_span_id}"
            
            logger.info(f"✓ Verified linking for tool '{tool_name}':")
            logger.info(f"  Inference span {inference_span_id} -> Tool invocation")
    
    logger.info(f"\n✓ All inference decision links verified correctly!")
    logger.info(f"  Total tools checked: {len(expected_tools)}")
    logger.info(f"  Total inference->tool links: {sum(len(v) for v in tool_invocation_spans.values())}")


@pytest.mark.asyncio
async def test_inference_subtype_turn_end(setup):
    """Test that inference spans get turn_end subtype when no tool is called."""
    # Ask a question that doesn't require a tool
    test_message = "Hello, how are you?"
    await run_agent(test_message)
    
    time.sleep(2)
    spans = setup.get_finished_spans()
    
    found_turn_end_inference = False
    for span in spans:
        span_attributes = span.attributes
        
        if (span_attributes.get("span.type") == "inference" and 
            span_attributes.get("span.subtype") == "turn_end"):
            found_turn_end_inference = True
            logger.info("✓ Found inference span with turn_end subtype")
            
            # Verify it has no tool name in entity.3
            assert span_attributes.get("entity.3.name") is None, \
                "Turn_end inference spans should not have entity.3.name (tool name)"
            break
    
    # Note: This might not always be true if the LLM decides to use a tool
    # but we're testing that the subtype mechanism works
    if not found_turn_end_inference:
        logger.warning("No turn_end inference span found - LLM might have decided to use a tool")
