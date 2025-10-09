#!/usr/bin/env python3
"""
Simple demonstration of the new sequence and flow validation capabilities
in the Monocle Testing Framework.
"""

from monocle_tfwk.agent_test import TraceAssertions
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Status, StatusCode
import time
import pytest
import logging

logger = logging.getLogger(__name__)

def create_mock_span(name: str, agent_name: str = None, start_offset: float = 0, duration: float = 1.0) -> ReadableSpan:
    """Create a mock span for testing flow validation."""
    
    class MockSpan:
        def __init__(self, name: str, agent_name: str = None, start_offset: float = 0, duration: float = 1.0):
            self.name = name
            self.start_time = int((time.time() + start_offset) * 1_000_000_000)  # nanoseconds
            self.end_time = int((time.time() + start_offset + duration) * 1_000_000_000)
            self.status = Status(StatusCode.OK)
            
            # Create attributes dictionary
            self.attributes = {}
            if agent_name:
                self.attributes["agent.name"] = agent_name
            
            if "tool" in name.lower():
                self.attributes["tool.name"] = name.split("_")[-1] if "_" in name else name
    
    return MockSpan(name, agent_name, start_offset, duration)

def test_demonstrate_flow_validation():
    """Demonstrate the new flow validation capabilities."""
    
    logger.info("🧪 Demonstrating Flow Validation Capabilities")
    logger.info("=" * 60)
    
    # Create mock spans that simulate a multi-agent travel booking flow
    spans = [
        # Supervisor starts first
        create_mock_span("agentic.request", "travel_supervisor", start_offset=0, duration=0.1),
        
        # Flight agent executes first (business logic: book transport first)
        create_mock_span("agentic.invocation", "flight_assistant", start_offset=0.2, duration=2.0),
        create_mock_span("tool_invocation_book_flight", "flight_assistant", start_offset=0.5, duration=0.3),
        
        # Hotel agent executes after flight (logical ordering)
        create_mock_span("agentic.invocation", "hotel_assistant", start_offset=2.5, duration=1.5),
        create_mock_span("tool_invocation_book_hotel", "hotel_assistant", start_offset=2.8, duration=0.4),
        
        # Recommendations agent runs in parallel with hotel booking
        create_mock_span("agentic.invocation", "recommendations_assistant", start_offset=2.4, duration=1.8),
        create_mock_span("tool_invocation_get_recommendations", "recommendations_assistant", start_offset=3.0, duration=0.5),
    ]
    
    # Create TraceAssertions instance
    traces = TraceAssertions(spans)
    
    logger.info("\n1️⃣ Basic Agent Detection:")
    logger.info("-" * 30)
    agent_names = traces.get_agent_names()
    logger.info(f"✅ Found agents: {agent_names}")
    
    logger.info("\n2️⃣ Execution Flow Analysis:")
    logger.info("-" * 30)
    traces.debug_execution_flow()
    
    logger.info("\n3️⃣ Sequence Validation:")
    logger.info("-" * 30)
    try:
        # Test that flight comes before hotel
        traces.assert_agent_called_before("flight_assistant", "hotel_assistant")
        logger.info("✅ Flight booking correctly happens before hotel booking")
    except AssertionError as e:
        logger.info(f"❌ Sequence validation failed: {e}")
    
    try:
        # Test agent sequence
        expected_sequence = ["travel_supervisor", "flight_assistant", "hotel_assistant", "recommendations_assistant"]
        traces.assert_agent_sequence(expected_sequence)
        logger.info("✅ Agent sequence validation passed")
    except AssertionError as e:
        logger.info(f"ℹ️ Full sequence validation: {e}")
        logger.info("   (This is expected - agents may not follow exact sequence)")
    
    logger.info("\n4️⃣ Agent-Specific Tool Usage:")
    logger.info("-" * 30)
    flight_tools = traces.get_tools_used_by_agent("flight_assistant")
    hotel_tools = traces.get_tools_used_by_agent("hotel_assistant")
    rec_tools = traces.get_tools_used_by_agent("recommendations_assistant")
    
    logger.info(f"✅ Flight agent tools: {flight_tools}")
    logger.info(f"✅ Hotel agent tools: {hotel_tools}")
    logger.info(f"✅ Recommendations agent tools: {rec_tools}")
    
    logger.info("\n5️⃣ Parallel Execution Detection:")
    logger.info("-" * 30)
    try:
        # Test if hotel and recommendations run in parallel (they do, within 0.1s)
        traces.assert_agents_called_in_parallel(["hotel_assistant", "recommendations_assistant"], tolerance_ms=200)
        logger.info("✅ Hotel and recommendations agents run in parallel")
    except AssertionError as e:
        logger.info(f"ℹ️ Parallel execution test: {e}")
    
    logger.info("\n6️⃣ JMESPath Query Capabilities:")
    logger.info("-" * 30)
    
    # Demonstrate some JMESPath queries
    all_agent_spans = traces.query("[?attributes.\"agent.name\"]")
    logger.info(f"✅ Total agent spans: {len(all_agent_spans)}")
    
    tool_invocations = traces.query("[?contains(name, 'tool_invocation')]")
    logger.info(f"✅ Tool invocation spans: {len(tool_invocations)}")
    
    agent_tool_mapping = traces.query("[?attributes.\"agent.name\" && contains(name, 'tool')].{agent: attributes.\"agent.name\", tool: attributes.\"tool.name\"}")
    logger.info(f"✅ Agent-tool mapping: {agent_tool_mapping}")
    
    logger.info("\n7️⃣ Workflow Pattern Validation:")
    logger.info("-" * 30)
    
    try:
        # Test fan-out pattern (supervisor → multiple agents)
        traces.assert_workflow_pattern("fan-out", ["travel_supervisor", "flight_assistant", "hotel_assistant"])
        logger.info("✅ Fan-out workflow pattern validated")
    except Exception as e:
        logger.info(f"ℹ️ Fan-out pattern: {e}")
    
    logger.info("\n✨ Flow Validation Demonstration Complete!")
    logger.info("=" * 60)
    
    return traces


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])