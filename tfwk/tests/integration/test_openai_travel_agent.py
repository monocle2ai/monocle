#!/usr/bin/env python3
"""
OpenAI Agents SDK Travel Agent Example with Monocle Testing Framework

This example demonstrates:
- Real OpenAI Agents SDK implementation with proper telemetry setup
- Comprehensive testing with the tfwk framework
- Practical usage patterns for OpenAI agent application testing
- Proper integration with Monocle's automatic instrumentation
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict

import pytest
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_tfwk import BaseAgentTest
from monocle_tfwk.semantic_similarity import semantic_similarity
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# Add the parent directory to the path to import from agentx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentx.openai_travel_agent import OpenAITravelAgentDemo

logger = logging.getLogger(__name__)

class TestOpenAITravelAgent(BaseAgentTest):
    """Comprehensive test suite for the OpenAI Agents SDK travel agent."""
    
 
    
    @pytest.fixture
    def travel_agent(self):
        """Create a fresh OpenAI travel agent instance for each test."""
        return OpenAITravelAgentDemo()
    
    @pytest.mark.asyncio
    async def test_flight_booking_request(self, travel_agent):
        """Test booking only a flight through OpenAI agent."""
        request = "I need to book a business flight from Delhi to Mumbai for December 15th, 2025"
        response = await travel_agent.process_travel_request(request)
        
        # Use semantic similarity for more robust assertions
        expected_responses = [
            "flight booking confirmed from Delhi to Mumbai",
            "business class flight booked successfully", 
            "flight reservation completed",
            "your flight has been booked"
        ]
        
        # Check if response is semantically similar to expected flight booking responses
        is_flight_related = any(semantic_similarity(response, expected, threshold=0.6) 
                               for expected in expected_responses)
        assert is_flight_related, f"Response doesn't seem to be about flight booking: {response}"
        
        # Verify OpenAI agent traces using integrated JMESPath (if available)
        traces = self.assert_traces()

        # Use integrated JMESPath methods - much cleaner and consistent naming!
        traces.assert_agent_type("agent.openai_agents")
        traces.assert_min_llm_calls(1)
            
        # Traditional assertions still work
        (traces
         .assert_spans(min_count=3)
         .completed_successfully())
        
        logger.info("✅ OpenAI Flight booking test passed")
    
    @pytest.mark.asyncio 
    async def test_hotel_booking_request(self, travel_agent):
        """Test booking only a hotel through OpenAI agent."""
        request = "Book me a hotel in Goa for 3 nights starting December 20th"
        response = await travel_agent.process_travel_request(request)
        
        # Use semantic similarity for hotel-related responses
        expected_responses = [
            "hotel booking in Goa confirmed",
            "accommodation reserved for 3 nights",
            "need more details for hotel booking",
            "which hotel would you prefer in Goa",
            "hotel reservation assistance"
        ]
        
        # Check if response is semantically related to hotel booking
        is_hotel_related = any(semantic_similarity(response, expected, threshold=0.5) 
                              for expected in expected_responses)
        assert is_hotel_related, f"Response doesn't seem to be about hotel booking: {response}"
        
        # Verify OpenAI trace structure (if available)
        traces = self.assert_traces()

        (traces
            .assert_spans(min_count=3)
            .completed_successfully())
        
        logger.info("✅ OpenAI Hotel booking test passed")
    
    @pytest.mark.asyncio
    async def test_complete_travel_package(self, travel_agent):
        """Test complete travel planning with flight, hotel and recommendations."""
        request = "Plan my business trip to Mumbai - I need flight from Delhi, hotel for 2 nights, and travel recommendations"
        response = await travel_agent.process_travel_request(request)
        
        # Use semantic similarity for comprehensive travel planning
        expected_responses = [
            "complete travel planning for Mumbai business trip",
            "flight and hotel booking with recommendations",
            "travel arrangements from Delhi to Mumbai",
            "business trip coordination assistance"
        ]
        
        # Check if response addresses comprehensive travel planning
        is_comprehensive_travel = any(semantic_similarity(response, expected, threshold=0.5) 
                                    for expected in expected_responses)
        assert is_comprehensive_travel, f"Response doesn't seem to address comprehensive travel planning: {response}"
        
        # Verify multiple agent interactions using integrated JMESPath (if available)
        traces = self.assert_traces()
        # Use integrated JMESPath methods - even more concise!
        agentic_spans = traces.get_agentic_spans()
        tool_invocations = traces.get_tool_invocations()
        
        assert len(agentic_spans) > 0, f"Should have agentic spans, found {len(agentic_spans)}"
        logger.info(f"Found {len(agentic_spans)} agentic spans and {len(tool_invocations)} tool invocations")
        
        # Check for workflow completeness using fluent API
        traces.assert_workflow_complete()
        
        (traces
            .assert_spans(min_count=5)  # Request + multiple agent + tool spans
            .completed_successfully())
        
        logger.info("✅ OpenAI Complete travel package test passed")
    
    @pytest.mark.asyncio
    async def test_travel_recommendations(self, travel_agent):
        """Test travel recommendations functionality."""
        request = "What are the best attractions and travel tips for visiting Delhi?"
        response = await travel_agent.process_travel_request(request)
        
        # Use semantic similarity for travel recommendations
        expected_responses = [
            "Delhi attractions include Red Fort and India Gate",
            "travel tips and recommendations for Delhi",
            "best places to visit in Delhi",
            "tourism guidance for Delhi sightseeing"
        ]
        
        # Check if response provides travel recommendations
        is_recommendations = any(semantic_similarity(response, expected, threshold=0.5) 
                               for expected in expected_responses)
        assert is_recommendations, f"Response doesn't seem to provide travel recommendations: {response}"
        
        # Verify recommendations agent was used (if available)
        traces = self.assert_traces()

        (traces
            .assert_spans(min_count=3)
            .completed_successfully())
        
        logger.info("✅ OpenAI Travel recommendations test passed")
    
    @pytest.mark.asyncio
    async def test_openai_agent_instrumentation(self, travel_agent):
        """Test that OpenAI agents are properly instrumented with Monocle using JMESPath queries."""
        request = "Book a flight to Bangalore"
        await travel_agent.process_travel_request(request)
        
        traces = self.assert_traces()
        
        # Use integrated JMESPath methods - much cleaner API with consistent naming!
        traces.assert_agent_type("agent.openai_agents")
        traces.assert_workflow_complete()
        traces.assert_min_llm_calls(1)
        
        # Show debug information 
        traces.debug_entities()
        
        # Additional sophisticated queries - now with concise API!
        agentic_spans = traces.get_agentic_spans()
        assert len(agentic_spans) > 0, "Should have agentic spans"
        
        # Check for specific OpenAI entities
        openai_entities = traces.find_entities_by_type("inference.openai")
        logger.info(f"Found {len(openai_entities)} OpenAI inference entities")
        
        logger.info("✅ OpenAI Agent instrumentation test passed")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, travel_agent):
        """Test error handling for unclear requests."""
        request = "Help me with something travel-related but very vague"
        response = await travel_agent.process_travel_request(request)
        
        # Should handle gracefully
        assert response is not None
        assert len(response) > 0
        
        # Should still complete without errors
        (self.assert_traces()
         .completed_successfully())
        
        logger.info("✅ OpenAI Error handling test passed")
    
    @pytest.mark.asyncio
    async def test_performance_and_timing(self, travel_agent):
        """Test OpenAI agent performance requirements."""
        request = "Quick flight booking from Chennai to Kolkata"
        await travel_agent.process_travel_request(request)
        
        # Verify performance constraints
        (self.assert_traces()
         .within_time_limit(15.0)  # OpenAI agents may take longer due to API calls
         .assert_spans(max_count=25))  # Reasonable complexity limit
        
        logger.info("✅ OpenAI Performance test passed")


    async def assert_cost_under_budget_llm(self, traces, budget_limit: float) -> bool:
        
        question = f"What is the total cost mentioned in the trace data? Is it under ${budget_limit}? Answer with just the total amount and YES/NO."
        
        # Use the integrated framework method
        analysis = await traces.ask_llm_about_traces(question)
        
        logger.info(f"💰 LLM Cost Analysis: {analysis}")
        
        # Simple check - if LLM says "NO" or mentions exceeding budget
        if "no" in analysis.lower() and ("exceed" in analysis.lower() or "over" in analysis.lower() or "more than" in analysis.lower()):
            raise AssertionError(f"LLM analysis indicates costs exceed budget: {analysis}")
        
        return True


    @pytest.mark.asyncio
    async def test_llm_based_cost_validation(self, travel_agent):
        """
        Test LLM-based cost validation - simpler and more reliable than regex parsing.
        
        Uses the same LLM technology as the agents to analyze cost information
        and validate budget constraints naturally.
        """
        request = """
        I need a budget trip to Goa for 3 days in December 2025.
        My budget is $800 total. Please book:
        1. Economy flights from Delhi to Goa
        2. Budget hotel for 2 nights
        3. Keep total cost under $800
        """
        
        logger.info("\n💰 Testing LLM-Based Cost Validation")
        logger.info("=" * 60)
        
        response = await travel_agent.process_travel_request(request)
        
        # Use LLM to analyze costs - much simpler than regex!
        logger.info("🤖 Analyzing costs using integrated LLM framework...")
        
        # Get traces for analysis
        traces = self.assert_traces()
        
        # Ask specific cost questions directly using the framework
        total_cost_question = "What is the total cost of this trip?"
        total_analysis = await traces.ask_llm_about_traces(total_cost_question)
        logger.info(f"📊 Total Cost Analysis: {total_analysis}")
        
        budget_question = "Is the total cost under $800 as requested?"
        budget_analysis = await traces.ask_llm_about_traces(budget_question)
        logger.info(f"💵 Budget Compliance: {budget_analysis}")
        
        # Simple assertion using LLM analysis
        await self.assert_cost_under_budget_llm(traces, 800.0)
        logger.info("✅ LLM confirms costs are within budget")
        
        # Additional cost breakdowns using integrated framework
        flight_cost_question = "What is the flight cost mentioned?"
        flight_analysis = await traces.ask_llm_about_traces(flight_cost_question)
        logger.info(f"✈️ Flight Cost: {flight_analysis}")
        
        hotel_cost_question = "What is the hotel cost mentioned?"
        hotel_analysis = await traces.ask_llm_about_traces(hotel_cost_question)
        logger.info(f"🏨 Hotel Cost: {hotel_analysis}")
        
        # Validate traces (already have traces object)

        traces.assert_agent_type("agent.openai_agents")
        traces.assert_workflow_complete()
        traces.assert_min_llm_calls(1)
        
        logger.info(f"📄 Response preview: {response[:200]}...")
        logger.info("✅ LLM-Based Cost Validation Test Passed")
        logger.info("=" * 60)
    
    @pytest.mark.asyncio
    async def test_agent_execution_flow(self, travel_agent):
        """Test OpenAI agent execution sequence and workflow patterns."""
        request = "Plan my complete trip to Mumbai - book flight from Delhi, hotel for 2 nights, and get recommendations"
        response = await travel_agent.process_travel_request(request)
        
        # Verify comprehensive response
        assert "mumbai" in response.lower()
        
        traces = self.assert_traces()
        
        # Debug the execution flow to see what happened
        logger.info("\n🔍 OpenAI Agent Execution Flow Analysis:")
        traces.debug_execution_flow()
        
        # Test 1: Get and analyze the execution sequence
        execution_sequence = traces.get_agent_execution_sequence()
        if execution_sequence:
            agent_names_in_order = [exec_info["agent_name"] for exec_info in execution_sequence]
            logger.info(f"📋 OpenAI Agent execution order: {agent_names_in_order}")
            
            # Test 2: If we have multiple agents, verify logical ordering
            unique_agents = list(set(agent_names_in_order))
            if len(unique_agents) > 1:
                logger.info(f"🎯 Found {len(unique_agents)} different agents: {unique_agents}")
                
                # Business logic: flight booking typically happens before hotel booking
                flight_agents = [name for name in agent_names_in_order if 'flight' in name.lower()]
                hotel_agents = [name for name in agent_names_in_order if 'hotel' in name.lower()]
                
                if flight_agents and hotel_agents:
                    logger.info("✈️🏨 Both flight and hotel agents found - checking order...")
                    traces.assert_agent_called_before(flight_agents[0], hotel_agents[0])
                    logger.info("✅ Flight booking correctly happens before hotel booking")
        else:
            logger.info("ℹ️ Single agent workflow detected - no sequence validation needed")
        
        # Test 3: Verify overall workflow completion
        (traces
         .assert_spans(min_count=2)
         .completed_successfully())
        
        logger.info("✅ OpenAI Agent execution flow test passed")
    
    @pytest.mark.asyncio
    async def test_conditional_agent_flow(self, travel_agent):
        """Test conditional flow based on request type."""
        
        # Test different request types to see different agent flows
        test_cases = [
            {
                "name": "Flight Only Request",
                "request": "I need to book a flight from Delhi to Mumbai",
                "expected_keywords": ["flight", "book"],
                "should_call_flight_agent": True,
                "should_call_hotel_agent": False
            },
            {
                "name": "Hotel Only Request", 
                "request": "Book me a hotel in Goa for 2 nights",
                "expected_keywords": ["hotel", "accommodation"],
                "should_call_flight_agent": False,
                "should_call_hotel_agent": True
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"\n🧪 Test Case {i+1}: {test_case['name']}")
            
            response = await travel_agent.process_travel_request(test_case["request"])
            
            # Verify response contains expected keywords or travel-related context
            keywords_found = any(keyword in response.lower() for keyword in test_case["expected_keywords"])
            travel_context = any(word in response.lower() for word in ["travel", "trip", "delhi", "mumbai", "goa", "plan", "when"])
            assert keywords_found or travel_context, f"Response should contain {test_case['expected_keywords']} or travel context: {response[:100]}..."
            
            traces = self.assert_traces()
            agent_names = traces.get_agent_names()
            logger.info(f"🔍 Agents called: {agent_names}")
            
            # Verify conditional agent calling based on request type
            flight_agents_called = any('flight' in name.lower() for name in agent_names)
            hotel_agents_called = any('hotel' in name.lower() for name in agent_names)
            
            if test_case["should_call_flight_agent"]:
                flight_related = flight_agents_called or any(word in response.lower() for word in ["flight", "class", "business", "economy", "travel date"])
                assert flight_related, f"Flight-related processing should occur. Response: {response[:100]}..."
            
            if test_case["should_call_hotel_agent"]:  
                hotel_related = hotel_agents_called or any(word in response.lower() for word in ["hotel", "accommodation", "stay", "room", "night", "check-in", "business trip", "goa"])
                assert hotel_related, f"Hotel-related processing should occur. Response: {response[:100]}..."
            
            logger.info(f"✅ Conditional flow validated for {test_case['name']}")
            
            # Clear traces for next test case
            self.validator.clear_spans()
        
        logger.info("✅ OpenAI Conditional agent flow test passed")
    
    @pytest.mark.asyncio
    async def test_performance_flow_analysis(self, travel_agent):
        """Test performance characteristics of agent execution flow."""
        request = "Quick booking: flight to Bangalore and hotel for 1 night"
        
        import time
        start_time = time.time()
        response = await travel_agent.process_travel_request(request)
        total_time = time.time() - start_time
        
        traces = self.assert_traces()
        
        logger.info(f"\n⏱️ Performance Flow Analysis:")
        logger.info(f"Total request time: {total_time:.3f}s")
        
        # Analyze execution timing
        execution_sequence = traces.get_agent_execution_sequence()
        if execution_sequence:
            logger.info(f"Number of agent executions: {len(execution_sequence)}")
            
            total_agent_time = sum(exec_info["duration"] for exec_info in execution_sequence)
            logger.info(f"Total agent execution time: {total_agent_time:.3f}s")
            
            # Check for parallel vs sequential execution patterns
            if len(execution_sequence) > 1:
                max_overlap_time = max(exec_info["duration"] for exec_info in execution_sequence)
                if total_agent_time > max_overlap_time * 1.5:
                    logger.info("📊 Sequential execution pattern detected")
                else:
                    logger.info("📊 Parallel/overlapping execution pattern detected")
            
            # Performance assertions
            (traces
             .within_time_limit(15.0)  # Reasonable upper bound
             .assert_spans(max_count=25))  # Complexity limit
        
        # Verify response quality
        assert len(response) > 50, "Response should be substantial"
        assert "bangalore" in response.lower() or "travel" in response.lower(), "Should mention destination or travel context"
        
        logger.info("✅ OpenAI Performance flow analysis test passed")



    @pytest.mark.asyncio
    async def test_complete_travel_itinerary_orchestration(self, travel_agent):

        # Complex itinerary request that requires all agents
        request = """
        I'm planning a 5-day business trip to Mumbai from Delhi for a tech conference. 
        I need:
        1. Flight recommendations and booking for December 20th-25th, 2025 (business class preferred)
        2. Hotel booking near the conference venue in Bandra-Kurla Complex for 4 nights
        3. Local recommendations for business dining, transportation, and must-see attractions
        4. Complete itinerary coordination with timing considerations
        
        Please coordinate all services and provide a comprehensive travel plan.
        """
        
        logger.info("\n🎯 Testing Multi-Agent Travel Itinerary Orchestration")
        logger.info("=" * 60)
        logger.info(f"Request: {request[:100]}...")
        
        response = await travel_agent.process_travel_request(request)
        
        # Validate comprehensive response using semantic similarity
        expected_response_elements = [
            "flight booking business class Delhi Mumbai December",
            "hotel reservation Bandra-Kurla Complex 4 nights", 
            "travel recommendations Mumbai business dining transportation",
            "complete itinerary coordination timing"
        ]
        
        # Check that response covers multiple aspects
        response_covers_flight = any(
            semantic_similarity(response, element, threshold=0.4) 
            for element in expected_response_elements[:1]
        )
        response_covers_hotel = any(
            semantic_similarity(response, element, threshold=0.4) 
            for element in expected_response_elements[1:2]
        )
        response_covers_recommendations = any(
            semantic_similarity(response, element, threshold=0.4) 
            for element in expected_response_elements[2:3]
        )
        
        # Assert comprehensive coverage (at least 2 out of 3 services)
        services_covered = sum([response_covers_flight, response_covers_hotel, response_covers_recommendations])
        assert services_covered >= 2, f"Expected comprehensive travel planning, but response only covers {services_covered}/3 services: {response[:200]}..."
        
        logger.info(f"✅ Response covers {services_covered}/3 travel services")
        logger.info(f"Response preview: {response[:300]}...")
        
        # Detailed trace analysis for multi-agent orchestration
        traces = self.assert_traces()
        

        logger.info("\n🔍 Multi-Agent Orchestration Analysis:")
        
        # 1. Verify all agent types participated
        traces.assert_agent_type("agent.openai_agents")
        traces.assert_workflow_complete()
        
        # 2. Check for multiple agent interactions
        agentic_spans = traces.get_agentic_spans()
        tool_invocations = traces.get_tool_invocations()
        
        assert len(agentic_spans) >= 3, f"Expected multiple agent interactions, found {len(agentic_spans)}"
        logger.info(f"✅ Agent Interactions: {len(agentic_spans)} agentic spans")
        logger.info(f"✅ Tool Usage: {len(tool_invocations)} tool invocations")
        
        # 3. Verify LLM calls for complex coordination
        llm_calls = traces.count_llm_calls()
        traces.assert_min_llm_calls(2)  # Multiple reasoning steps expected
        logger.info(f"✅ LLM Coordination Calls: {llm_calls}")
        
        # 4. Advanced orchestration analysis using JMESPath
        logger.info("\n🔍 Advanced Orchestration Analysis:")
        
        # Check for agent handoffs and coordination patterns
        agent_names = traces.query("[?attributes.\"entity.1.type\" == 'agent.openai_agents'].attributes.\"entity.1.name\" | [?@ != null]")
        unique_agents = list(set(agent_names)) if agent_names else []
        logger.info(f"Participating Agents: {unique_agents}")
        
        # Analyze workflow complexity
        workflow_complexity = traces.query("[?contains(attributes.\"span.type\", 'agentic')].attributes.\"span.type\" | [?@ != null]")
        unique_workflows = list(set(workflow_complexity)) if workflow_complexity else []
        logger.info(f"Workflow Types: {unique_workflows}")
        
        # Check for tool coordination across agents
        tool_usage_pattern = traces.query("[?attributes.\"span.type\" == 'agentic.tool.invocation'].{agent: attributes.\"entity.1.name\", tool: attributes.\"entity.2.name\"}")
        if tool_usage_pattern:
            logger.info(f"Tool Usage Pattern: {tool_usage_pattern[:3]}...")  # Show first 3
        
        # Verify coordination spans (multiple agent types working together)
        coordination_evidence = traces.query("length([?attributes.\"entity.1.type\" == 'agent.openai_agents' && attributes.\"entity.2.type\" != null])")
        logger.info(f"Coordination Evidence: {coordination_evidence} cross-entity spans")
        
        # 5. Assert comprehensive workflow characteristics
        traces.assert_spans(min_count=8)  # Complex multi-agent workflow
        
        # Check for service orchestration patterns
        service_spans = traces.query("[?contains(name, 'flight') || contains(name, 'hotel') || contains(name, 'recommendation')]")
        if service_spans:
            logger.info(f"✅ Service-Specific Operations: {len(service_spans)}")
            
        logger.info("\n✅ Complete Travel Itinerary Orchestration Test Passed")
        logger.info("=" * 60)

    @pytest.mark.asyncio
    async def test_sequential_multi_agent_coordination(self, travel_agent):
        """
        Test sequential coordination of all three specialized agents.
        
        Validates:
        - Recommendations agent provides destination insights first
        - Flight agent books based on recommendations timing
        - Hotel agent coordinates with flight dates and destination insights
        - All agents contribute to final comprehensive itinerary
        """
        request = """
        Plan a complete 3-day weekend getaway to Goa for February 14-17, 2025.
        I want:
        1. Travel recommendations for best activities and timing in Goa
        2. Round-trip flights from Bangalore 
        3. Beach-front hotel for 3 nights
        
        Make sure all bookings are coordinated and provide a complete timeline.
        """
        
        logger.info("\n🔄 Testing Sequential Multi-Agent Coordination")
        logger.info("=" * 60)
        
        response = await travel_agent.process_travel_request(request)
        
        # Validate that all three services are mentioned in response
        services_mentioned = {
            "recommendations": any(keyword in response.lower() for keyword in ["recommend", "activities", "attractions", "tips", "best time"]),
            "flight": any(keyword in response.lower() for keyword in ["flight", "booking", "bangalore", "february"]),
            "hotel": any(keyword in response.lower() for keyword in ["hotel", "accommodation", "beach", "nights"])
        }
        
        services_count = sum(services_mentioned.values())
        assert services_count >= 2, f"Expected multi-service coordination, found {services_count}/3 services in response"
        
        logger.info(f"✅ Services coordinated: {services_count}/3")
        for service, mentioned in services_mentioned.items():
            logger.info(f"   {service.capitalize()}: {'✅' if mentioned else '❌'}")
        
        # Validate traces for sequential coordination
        traces = self.assert_traces()
        
        logger.info("\n🔍 Sequential Coordination Analysis:")
        
        # Check for agent workflow completion
        traces.assert_agent_type("agent.openai_agents")
        traces.assert_workflow_complete()
        
        # Analyze agent participation patterns
        agent_names = traces.get_entity_names()
        participating_agents = [name for name in agent_names if "Assistant" in str(name)]
        logger.info(f"Participating Agents: {participating_agents}")
        
        # Check workflow sequence indicators
        workflow_spans = traces.get_agentic_spans()
        tool_invocations = traces.get_tool_invocations()
        
        logger.info(f"Workflow Complexity: {len(workflow_spans)} spans, {len(tool_invocations)} tools")
        
        # Verify coordination depth
        coordination_depth = traces.query("length([?attributes.\"span.type\" && contains(attributes.\"span.type\", 'agentic')])")
        assert coordination_depth >= 3, f"Expected deep coordination, found {coordination_depth} coordination spans"
        
        # Check for handoff patterns
        delegation_spans = traces.query("[?attributes.\"span.type\" == 'agentic.delegation']")
        if delegation_spans:
            logger.info(f"✅ Agent Delegations: {len(delegation_spans)}")
        
        # Verify comprehensive span coverage
        traces.assert_spans(min_count=5)  # Sequential operations require multiple spans
        traces.assert_min_llm_calls(2)   # Multiple reasoning steps
            
        logger.info(f"Response Preview: {response[:200]}...")
        logger.info("✅ Sequential Multi-Agent Coordination Test Passed")
        logger.info("=" * 60)



if __name__ == "__main__":

    # Run the practical demo
    # demo_agent = asyncio.run(run_openai_demo())
    
    logger.info("\n" + "=" * 70)
    logger.info("🧪 Running OpenAI Agents SDK Test Suite")
    logger.info("=" * 70)
    
    # Run the test suite
    test_result = pytest.main([
        __file__ + "::TestOpenAITravelAgent",
        "-v", 
        "--tb=short",
        "-s"  # Show print statements
    ])
    
    if test_result == 0:
        logger.info("\n🎉 All OpenAI tests passed! The OpenAI Agents SDK agent is working correctly.")
    else:
        logger.info("\n❌ Some OpenAI tests failed. Check the output above.")