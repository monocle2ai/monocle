#!/usr/bin/env python3
"""
Advanced JMESPath Query Examples for Monocle Testing Framework

This test demonstrates sophisticated trace analysis using JMESPath queries
with the integrated TraceAssertions API. It showcases how to perform
complex data extraction and analysis on agent traces.
"""

import asyncio

import pytest
from agentx.openai_travel_agent import OpenAITravelAgentDemo
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_tfwk import BaseAgentTest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class TestJMESPathAdvancedQueries(BaseAgentTest):
    """Advanced JMESPath query demonstrations for trace analysis."""
    
    @pytest.fixture
    def travel_agent(self):
        """Create a fresh OpenAI travel agent instance for each test."""
        return OpenAITravelAgentDemo()
    
    @pytest.mark.asyncio
    async def test_jmespath_advanced_queries(self, travel_agent):
        """Demonstrate advanced JMESPath queries for trace analysis."""
        request = "Plan my complete trip to Delhi with flight, hotel, and recommendations"
        await travel_agent.process_travel_request(request)
        
        traces = self.assert_traces()
        
        if travel_agent.agents_available:
            print("\n🔍 Advanced JMESPath Query Demonstrations:")
            
            # Advanced entity analysis using integrated query method
            agent_entities = traces.query("[?attributes.\"entity.1.type\" == 'agent.openai_agents'].attributes.\"entity.1.name\"")
            print(f"Agent Names: {agent_entities}")
            
            # Complex span filtering
            inference_spans = traces.query("[?attributes.\"span.type\" == 'inference'].{name: name, model: attributes.\"entity.2.name\", endpoint: attributes.\"entity.1.inference_endpoint\"}")
            print(f"Inference Details: {inference_spans}")
            
            # Workflow analysis
            workflow_spans = traces.query("[?contains(attributes.\"span.type\", 'agentic')].{type: attributes.\"span.type\", subtype: attributes.\"span.subtype\", agent: attributes.\"entity.1.name\"}")
            print(f"Workflow Spans: {workflow_spans}")
            
            # Entity counting by type
            entity_counts = traces.query("group_by([].attributes.\"entity.1.type\", &@) | map(&{type: [0], count: length([1])}, @)")
            print(f"Entity Type Counts: {entity_counts}")
            
            # Performance analysis (if timing data available)
            spans_with_timing = traces.query("[?start_time && end_time].{name: name, duration_available: `true`}")
            print(f"Spans with timing: {len(spans_with_timing) if spans_with_timing else 0}")
            
        print("✅ JMESPath advanced queries test passed")
    
    @pytest.mark.asyncio
    async def test_jmespath_entity_extraction_patterns(self, travel_agent):
        """Demonstrate various JMESPath patterns for entity extraction."""
        request = "Book a flight and hotel for my business trip to Mumbai"
        await travel_agent.process_travel_request(request)
        
        traces = self.assert_traces()
        
        if travel_agent.agents_available:
            print("\n🔍 JMESPath Entity Extraction Patterns:")
            
            # Pattern 1: Extract all entity types across all positions
            all_entity_types = traces.query("[].[attributes.\"entity.1.type\", attributes.\"entity.2.type\", attributes.\"entity.3.type\"] | [] | [?@ != null]")
            print(f"All Entity Types: {set(all_entity_types) if all_entity_types else []}")
            
            # Pattern 2: Find spans with specific attribute patterns
            llm_model_spans = traces.query("[?attributes.\"entity.1.type\" == 'inference.openai'].{span_name: name, model: attributes.\"entity.2.name\"}")
            print(f"LLM Model Usage: {llm_model_spans}")
            
            # Pattern 3: Complex filtering with multiple conditions
            agent_tool_spans = traces.query("[?attributes.\"span.type\" == 'agentic.tool.invocation' && attributes.\"entity.1.type\" == 'agent.openai_agents']")
            print(f"Agent Tool Invocations: {len(agent_tool_spans) if agent_tool_spans else 0}")
            
            # Pattern 4: Nested data extraction
            workflow_summary = traces.query("[?contains(attributes.\"span.type\", 'agentic')].{workflow_type: attributes.\"span.type\", agent_name: attributes.\"entity.1.name\", operation: attributes.\"span.subtype\"} | [?workflow_type != null]")
            print(f"Workflow Summary: {workflow_summary}")
            
        print("✅ JMESPath entity extraction patterns test passed")
    
    @pytest.mark.asyncio
    async def test_jmespath_performance_analysis(self, travel_agent):
        """Demonstrate JMESPath queries for performance and timing analysis."""
        request = "I need travel recommendations and booking assistance"
        await travel_agent.process_travel_request(request)
        
        traces = self.assert_traces()
        
        if travel_agent.agents_available:
            print("\n🔍 JMESPath Performance Analysis:")
            
            # Pattern 1: Count operations by type
            span_type_counts = traces.query("group_by([].attributes.\"span.type\", &@) | map(&{type: [0], count: length([1])}, @) | [?type != null]")
            print(f"Span Type Distribution: {span_type_counts}")
            
            # Pattern 2: Find the longest span names (complexity indicator)
            long_span_names = traces.query("[].name | [?length(@) > `20`] | sort(@) | reverse(@)")
            print(f"Complex Operation Names: {long_span_names[:3] if long_span_names else []}")
            
            # Pattern 3: Entity relationship analysis
            entity_relationships = traces.query("[].{span: name, primary_entity: attributes.\"entity.1.type\", secondary_entity: attributes.\"entity.2.type\"} | [?primary_entity != null && secondary_entity != null]")
            print(f"Entity Relationships Found: {len(entity_relationships) if entity_relationships else 0}")
            
            # Pattern 4: Operation complexity scoring
            complex_operations = traces.query("[?attributes.\"entity.1.type\" && attributes.\"entity.2.type\" && attributes.\"entity.3.type\"]")
            print(f"High-Complexity Operations: {len(complex_operations) if complex_operations else 0}")
            
        print("✅ JMESPath performance analysis test passed")
    
    @pytest.mark.asyncio
    async def test_jmespath_debugging_helpers(self, travel_agent):
        """Demonstrate JMESPath queries useful for debugging and troubleshooting."""
        request = "Help me plan a vacation with flights and accommodation"
        await travel_agent.process_travel_request(request)
        
        traces = self.assert_traces()
        
        if travel_agent.agents_available:
            print("\n🔍 JMESPath Debugging Helpers:")
            
            # Pattern 1: Find all unique attribute keys (schema discovery)
            all_attr_keys = traces.query("[].attributes | [].keys(@) | [] | sort(@) | sort(@)")
            unique_keys = list(set(all_attr_keys)) if all_attr_keys else []
            print(f"Available Attribute Keys: {unique_keys[:10]}...")  # Show first 10
            
            # Pattern 2: Identify spans with missing expected attributes
            spans_missing_entity = traces.query("[?!attributes.\"entity.1.type\"].name")
            print(f"Spans Without Primary Entity: {spans_missing_entity[:3] if spans_missing_entity else []}")
            
            # Pattern 3: Find error indicators
            error_indicators = traces.query("[?contains(name, 'error') || contains(name, 'Error') || contains(name, 'fail')].name")
            print(f"Potential Error Indicators: {error_indicators}")
            
            # Pattern 4: Extract span hierarchy information
            root_spans = traces.query("[?!attributes.\"parent.id\"].name")
            print(f"Root-Level Spans: {root_spans}")
            
        print("✅ JMESPath debugging helpers test passed")


async def run_jmespath_demo():
    """Run a demonstration of advanced JMESPath queries."""
    print("🚀 Starting Advanced JMESPath Query Demonstrations")
    print("=" * 70)
    
    # Initialize telemetry
    memory_exporter = InMemorySpanExporter()
    span_processors = [SimpleSpanProcessor(memory_exporter)]
    
    instrumentor = setup_monocle_telemetry(
        workflow_name="jmespath_advanced_demo",
        span_processors=span_processors
    )
    
    try:
        # Create agent and run a complex operation
        agent = OpenAITravelAgentDemo()
        
        if not agent.agents_available:
            print("ℹ️  OpenAI Agents SDK not available - running with mock responses")
        
        print("\n🔍 Generating trace data for JMESPath analysis...")
        request = "Plan my business trip to Delhi - need flight, hotel, and local recommendations"
        response = await agent.process_travel_request(request)
        print(f"Agent Response: {response[:100]}...")
        
        print("\n✅ JMESPath demo data generated successfully!")
        print("Run the test suite to see detailed JMESPath query examples.")
        
    finally:
        if instrumentor and hasattr(instrumentor, 'is_instrumented_by_opentelemetry'):
            if instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.uninstrument()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(run_jmespath_demo())
    
    print("\n" + "=" * 70)
    print("🧪 Running Advanced JMESPath Test Suite")
    print("=" * 70)
    
    # Run the test suite
    test_result = pytest.main([
        __file__ + "::TestJMESPathAdvancedQueries",
        "-v", 
        "--tb=short",
        "-s"  # Show print statements
    ])
    
    if test_result == 0:
        print("\n🎉 All JMESPath advanced query tests passed!")
    else:
        print("\n❌ Some JMESPath tests failed. Check the output above.")