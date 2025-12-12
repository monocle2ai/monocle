#!/usr/bin/env python3
"""
OpenAI Agents SDK Travel Agent Example with Monocle Testing Framework

This example demonstrates:
- Real OpenAI Agents SDK implementation with proper telemetry setup
- Comprehensive testing with the tfwk framework
- Practical usage patterns for OpenAI agent application testing
- Proper integration with Monocle's automatic instrumentation
"""


import logging

import pytest
from agentx.openai_travel_agent import OpenAITravelAgentDemo
from monocle_tfwk import BaseAgentTest
from monocle_tfwk.schema import MonocleSpanType

logger = logging.getLogger(__name__)

class TestOpenAITravelAgent(BaseAgentTest):
    """Comprehensive test suite for the OpenAI Agents SDK travel agent."""
    
    @pytest.fixture
    def travel_agent(self):
        """Create a fresh OpenAI travel agent instance for each test."""
        return OpenAITravelAgentDemo()
   
    
    def get_flow(self):
        """
        Create a comprehensive participant registry for agentic flow validation.
        
        Returns:
            dict: Comprehensive flow definition with participants, interactions, and constraints
        """
        return {
            "participants": [
                # Format: (alias, name, type, description)
                ("U", "User", "actor", "Human user initiating travel requests"),
                ("TC", "Travel Coordinator", "agent", "Main orchestration agent"),
                ("FA", "Flight Assistant", "agent", "Flight booking specialist"),
                ("HA", "Hotel Assistant", "agent", "Hotel booking specialist"),
                ("RA", "Recommendations Assistant", "agent", "Travel recommendations provider"),
                ("GRT", "get_recommendations_tool", "tool", "Retrieves destination recommendations"),
                ("BFT", "book_flight_tool", "tool", "Executes flight bookings"),
                ("BHT", "book_hotel_tool", "tool", "Executes hotel bookings")
            ],
            "required": ["TC"],  # Minimum required participants (Travel Coordinator should always be present)
            #"forbidden": [("U", "FA"), ("U", "HA"), ("U", "RA")],  # Direct interactions not allowed
            "interactions": [
                # Match actual OpenAI Agents SDK span structure using MonocleSpanType enum values
                ("U", "TC", MonocleSpanType.AGENTIC_REQUEST),  # User invokes Travel Coordinator
                ("TC", "RA", MonocleSpanType.AGENTIC_DELEGATION),  # Travel Coordinator self-interaction
                ("RA", "GRT", MonocleSpanType.AGENTIC_TOOL_INVOCATION),  # Recommendations Assistant invocation
                ("TC", "FA", MonocleSpanType.AGENTIC_DELEGATION),         # TC delegates to FA (agentic.delegation)
                ("FA", "BFT", MonocleSpanType.AGENTIC_TOOL_INVOCATION),         # Flight Assistant using book_flight_tool
                ("TC", "HA", MonocleSpanType.AGENTIC_DELEGATION),          # TC delegates to HA  
                ("HA", "BHT", MonocleSpanType.AGENTIC_TOOL_INVOCATION),         # Hotel Assistant using book_hotel_tool
            ]
        }
    
    # === COMPREHENSIVE FLOW VALIDATION TESTS ===
    
    @pytest.mark.asyncio
    async def test_full_travel_booking_flow(self, travel_agent):
        """
        Test complete travel booking workflow with comprehensive flow validation.
        
        This test validates the entire agent orchestration pattern from user request
        through Travel Coordinator delegation to specialized agents and tool invocations.
        """
        logger.info("🧪 Testing full travel booking flow with agentic validation")
        
        # Complex travel request that should trigger multiple agent interactions
        travel_request = (
            "I need to plan a business trip to Mumbai. Please book a business class flight "
            "from Delhi to Mumbai on December 15th, 2025, recommend things to see, "
            "and book a luxury hotel for 2 nights starting December 15th."
        )
        
        # Process the travel request
        result = await travel_agent.process_travel_request(travel_request)
        
        # Basic assertions on response
        assert result and len(result) > 0, "Should receive non-empty response"
        assert any(keyword in result.lower() for keyword in ["flight", "hotel", "mumbai"]), \
            f"Response should mention travel elements: {result}"
        
        # === SIMPLIFIED AGENTIC FLOW VALIDATION BASED ON OBSERVED PATTERNS ===
        
        
        # Assert the realistic flow pattern
        (self.assert_traces()  # Get trace assertions
         .assert_agent_flow(self.get_flow())  # Use enhanced flow validation
        )
        
        logger.info("✅ Full travel booking flow validation passed")
    
    # @pytest.mark.asyncio
    # async def test_flight_booking_specific_flow(self, travel_agent):
    #     """
    #     Test flight booking specific workflow with targeted flow validation.
    #     """
    #     logger.info("🧪 Testing flight booking specific flow")
        
    #     flight_request = "Book a business class flight from Delhi to Mumbai on December 15th, 2025"
        
    #     result = await travel_agent.process_travel_request(flight_request)
        
    #     # Validate response content
    #     assert "flight" in result.lower(), f"Response should mention flight: {result}"
        
    #     # Validate specific flight booking interactions using actual OpenAI Agents SDK patterns
    #     flight_interactions = [
    #         ("Travel Coordinator", "Flight Assistant", "delegation"),  # Coordinator delegates to Flight Assistant
    #         ("Flight Assistant", "Travel Coordinator", "invocation")   # Flight Assistant invocation
    #     ]
        
    #     (self.assert_traces()
    #      .assert_agent_flow(flight_interactions)  # Legacy tuple format still supported
    #     )
        
    #     logger.info("✅ Flight booking flow validation passed")
        
    # @pytest.mark.asyncio 
    # async def test_hotel_booking_specific_flow(self, travel_agent):
    #     """
    #     Test hotel booking specific workflow with targeted flow validation.
    #     """
    #     logger.info("🧪 Testing hotel booking specific flow")
        
    #     hotel_request = "Book a luxury hotel in Goa for 3 nights starting December 20th"
        
    #     result = await travel_agent.process_travel_request(hotel_request)
        
    #     # Validate response content
    #     assert "hotel" in result.lower(), f"Response should mention hotel: {result}"
        
    #     # Validate specific hotel booking interactions using actual OpenAI Agents SDK patterns
    #     hotel_interactions = [
    #         ("Travel Coordinator", "Hotel Assistant", "delegation"),   # Coordinator delegates to Hotel Assistant
    #         ("Hotel Assistant", "Travel Coordinator", "invocation")    # Hotel Assistant invocation
    #     ]
        
    #     (self.assert_traces()
    #      .assert_agent_flow(hotel_interactions)
    #     )
        
    #     logger.info("✅ Hotel booking flow validation passed")
    
    # @pytest.mark.asyncio
    # async def test_recommendations_flow(self, travel_agent):
    #     """
    #     Test travel recommendations workflow with flow validation.
    #     """
    #     logger.info("🧪 Testing recommendations flow")
        
    #     recommendations_request = "What are the best attractions to visit in Delhi?"
        
    #     result = await travel_agent.process_travel_request(recommendations_request)
        
    #     # Validate response content
    #     assert any(keyword in result.lower() for keyword in ["attraction", "recommend", "delhi"]), \
    #         f"Response should mention recommendations: {result}"
        
    #     # Validate specific recommendations interactions using actual OpenAI Agents SDK patterns  
    #     recommendations_interactions = [
    #         ("Travel Coordinator", "Recommendations Assistant", "delegation"),     # Coordinator delegates to Recommendations Assistant
    #         ("Recommendations Assistant", "Travel Coordinator", "invocation")      # Recommendations Assistant invocation
    #     ]
        
    #     (self.assert_traces()
    #      .assert_agent_flow(recommendations_interactions)
    #     )
        
    #     logger.info("✅ Recommendations flow validation passed")
    

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