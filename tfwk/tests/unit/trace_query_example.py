#!/usr/bin/env python3
"""
Example showing how to use TraceQueryEngine from the monocle_tfwk framework.
"""

import logging

import pytest
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_tfwk import BaseAgentTest, TraceQueryEngine
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTraceQueryEngineUsage(BaseAgentTest):
    """Example test showing TraceQueryEngine usage from the framework."""
    
    @pytest.mark.asyncio
    async def test_simple_trace_query_example(self):
        """Demonstrate simple TraceQueryEngine usage."""
        # Simulate some test action that creates traces
        logger.info("Performing test action that creates traces")
        
        # Get traces
        traces = self.assert_traces()
        
        # Use TraceQueryEngine from the framework
        query_engine = TraceQueryEngine(traces)
        
        # Perform some basic queries
        print("\n🔍 TraceQueryEngine Framework Usage Example:")
        
        # Get all entity types
        entity_types = query_engine.get_all_entity_types()
        print(f"Entity Types Found: {entity_types}")
        
        # Get all entity names  
        entity_names = query_engine.get_entity_names()
        print(f"Entity Names Found: {entity_names}")
        
        # Count workflow spans
        workflow_spans = query_engine.get_agentic_spans()
        print(f"Workflow Spans: {len(workflow_spans)}")
        
        # Custom JMESPath query
        custom_result = query_engine.query("[].name")
        print(f"All Span Names: {custom_result}")
        
        # Debug output
        query_engine.debug_entities()
        
        print("✅ TraceQueryEngine framework usage example completed")


if __name__ == "__main__":
    # Initialize Monocle telemetry
    memory_exporter = InMemorySpanExporter()
    span_processors = [SimpleSpanProcessor(memory_exporter)]
    
    instrumentor = setup_monocle_telemetry(
        workflow_name="trace_query_example",
        span_processors=span_processors
    )
    
    try:
        print("🚀 Running TraceQueryEngine Framework Usage Example")
        print("=" * 60)
        
        # Run the test
        test_result = pytest.main([
            __file__ + "::TestTraceQueryEngineUsage::test_simple_trace_query_example",
            "-v", 
            "-s"
        ])
        
        if test_result == 0:
            print("\n🎉 TraceQueryEngine framework usage example completed successfully!")
        else:
            print("\n❌ Example failed.")
            
    finally:
        # Clean up instrumentation
        if instrumentor and hasattr(instrumentor, 'is_instrumented_by_opentelemetry'):
            if instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.uninstrument()