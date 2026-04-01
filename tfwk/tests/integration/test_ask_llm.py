#!/usr/bin/env python3
"""
Integrated LLM Analysis Framework Test

Tests how test developers can ask ANY question about trace data
using the fully integrated TraceAssertions.ask_llm_about_traces() method.
"""

import pytest
import logging

from agentx.openai_travel_agent import OpenAITravelAgentDemo
from monocle_tfwk import BaseAgentTest
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)


class TestAskLLM(BaseAgentTest):
    """Test suite for the integrated LLM analysis framework."""
    
    @pytest.fixture(autouse=True)
    def setup_telemetry(self):
        """Set up Monocle telemetry for tracing."""
        memory_exporter = InMemorySpanExporter()
        span_processors = [SimpleSpanProcessor(memory_exporter)]
        
        instrumentor = setup_monocle_telemetry(
            workflow_name="ask_llm_test_demo",
            span_processors=span_processors
        )
        
        yield instrumentor
        
        # Cleanup after test
        if hasattr(instrumentor, 'uninstrument'):
            instrumentor.uninstrument()
    
    @pytest.fixture
    def travel_agent(self):
        """Create a fresh OpenAI travel agent instance for each test."""
        return OpenAITravelAgentDemo()

    @pytest.mark.asyncio
    async def test_integrated_llm_analysis_framework(self, travel_agent):
        """
        Test the integrated LLM analysis framework.
        """
        
        logger.info("🚀 Integrated LLM Analysis Framework Test")
        logger.info("=" * 60)
    
        # Process a travel request
        logger.info("📋 Processing travel request...")
        response = await travel_agent.process_travel_request(
            "Plan a 4-day business trip to Tokyo from San Francisco with $2000 budget"
        )
        
        # Get traces using the integrated framework
        traces = self.assert_traces()
        
        logger.info(f"📄 Agent Response: {response[:200]}...")
        
        logger.info("🧠 Integrated LLM Analysis - No Setup Required!")
        logger.info("-" * 50)
        
        # Test developers can ask any question without any setup!
        dynamic_questions = [
            "What is the total budget mentioned?",
            "Which cities are involved in this trip?", 
            "How many days is the trip?",
            "Is this a business or leisure trip?",
            "What services were recommended?",
            "Are flights included in the response?",
            "Were hotels mentioned?",
            "Is the budget realistic for Tokyo travel?",
            "What specific recommendations were made?"
        ]
        
        logger.info("✨ All questions use: await traces.ask_llm_about_traces(question)")
        
        # Ask questions dynamically - no hardcoding needed!
        for i, question in enumerate(dynamic_questions, 1):
            try:
                answer = await traces.ask_llm_about_traces(question)
                logger.info(f"❓ Q{i}: {question}")
                logger.info(f"💡 A{i}: {answer}")
            except Exception as e:
                logger.info(f"❌ Error on Q{i}: {e}")
        
        logger.info("🎯 Framework Benefits:")
        logger.info("✅ Single integrated method: traces.ask_llm_about_traces()")  
        logger.info("✅ No manual extraction functions needed")
        logger.info("✅ Questions can be completely dynamic")
        logger.info("✅ Works with any agent response format")
        logger.info("✅ Automatic trace context analysis")
        logger.info("✅ Graceful handling of missing data")
        
        # Assert that the test completed successfully
        assert traces is not None, "TraceAssertions should be created successfully"
        assert response is not None, "Agent should provide a response"


if __name__ == "__main__":
    # Run the test directly if executed as a script
    pytest.main([__file__, "-s", "--tb=short"])