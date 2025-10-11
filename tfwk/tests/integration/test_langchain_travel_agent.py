#!/usr/bin/env python3
"""
LangChain Travel Agent Example with Monocle Testing Framework

This example demonstrates:
- Real LangChain/LangGraph implementation with proper telemetry setup
- Comprehensive testing with the tfwk framework
- Practical usage patterns for LangChain agent application testing
- Proper integration with Monocle's automatic instrumentation
"""

import logging
import sys
from pathlib import Path

import pytest

# Add the parent directory to the path to import from agentx
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentx.langchain_travel_agent import LangChainTravelAgentDemo
from monocle_tfwk import BaseAgentTest

logger = logging.getLogger(__name__)

class TestLangChainTravelAgent(BaseAgentTest):
    """Comprehensive test suite for the LangChain travel agent."""
    
    participants = [
                # Format: (alias, name, type, description)
                ("U", "User", "actor", "Human user initiating travel requests"),
                ("TC", "Travel_Coordinator", "agent", "Main orchestration agent"),
                ("FA", "Flight_Assistant", "agent", "Flight booking specialist"),  # Updated to match actual traces
                ("HA", "Hotel_Assistant", "agent", "Hotel booking specialist"),
                ("RA", "Recommendations_Assistant", "agent", "Travel recommendations provider"),
                ("GRT", "get_recommendations_tool", "tool", "Retrieves destination recommendations"),
                ("BFT", "book_flight_tool", "tool", "Executes flight bookings"),
                ("BHT", "book_hotel_tool", "tool", "Executes hotel bookings")
            ]

    @pytest.fixture
    def travel_agent(self):
        """Create a fresh LangChain travel agent instance for each test."""
        return LangChainTravelAgentDemo()
   
    # === COMPREHENSIVE FLOW VALIDATION TESTS ===
    
    @pytest.mark.asyncio
    async def test_full_travel_booking_flow(self, travel_agent):
        """
        Test complete travel booking workflow with comprehensive flow validation.
        
        This test validates the entire agent orchestration pattern from user request
        through Travel Coordinator delegation to specialized agents and tool invocations.
        """
        logger.info("🧪 Testing full travel booking flow with agentic validation")
        
        whole_flow = {
            "participants": self.participants,
            "required": ["TC"],  # Minimum required participants (Travel Coordinator should always be present)
            #"forbidden": [("U", "FA"), ("U", "HA"), ("U", "RA")],  # Direct interactions not allowed
            "interactions": [
                # Match actual LangChain/LangGraph span structure using aliases
                ("U", "TC", "request"),  # User invokes Travel Coordinator
                ("TC", "RA", "delegation"),         # TC delegates to RA
                ("RA", "GRT", "invocation"),         # Recommendations Assistant using get_recommendations_tool
                ("TC", "FA", "delegation"),         # TC delegates to FA (agentic.delegation)
                ("FA", "BFT", "invocation"),         # Flight Assistant using book_flight_tool
                ("TC", "HA", "delegation"),          # TC delegates to HA  
                ("HA", "BHT", "invocation"),         # Hotel Assistant using book_hotel_tool
            ]
        }

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
        traces = self.assert_traces()
        (traces  # Get trace assertions
         .assert_agent_flow(whole_flow)
        )

        # Test the JSON format confirmation
        confirmation = await traces.ask_llm_about_traces(
            "Is the hotel booking confirmed? "
            "Return your answer in JSON format with this exact structure: "
            '{"confirmed": true/false, "reason": "brief explanation"}'
        )
        
        # Extract and assert the confirmed field
        import json
        confirmation_data = json.loads(confirmation)
        assert confirmation_data["confirmed"], f"Hotel booking should be confirmed: {confirmation}"
        logger.info(f"✅ Full travel booking flow validation passed {confirmation}")
        


    # @pytest.mark.asyncio
    async def test_flight_booking_specific_flow(self, travel_agent):
        """
        Test flight booking specific workflow with targeted flow validation.
        """
        logger.info("🧪 Testing flight booking specific flow")
        flight_request = "Book a business class flight from Delhi to Mumbai on December 15th, 2025"
        result = await travel_agent.process_travel_request(flight_request)
        
        # Validate response content
        assert "flight" in result.lower(), f"Response should mention flight: {result}"
        
        # Validate specific flight booking interactions using actual LangChain/LangGraph patterns
        flight_interactions = {
            "participants": self.participants,
            "interactions": [
                ("TC", "FA", "delegation"),  # Coordinator delegates to Flight Assistant
                ("FA", "BFT", "invocation")   # Flight Assistant invocation
            ]
        }
        (self.assert_traces()
         .assert_agent_flow(flight_interactions)  # Legacy tuple format still supported
        )
        logger.info("✅ Flight booking flow validation passed")

    #TODO: The following test can fail as sometimes the ("HA", "BHT", "invocation") is missing
    # in the traces json. This requires the instrumentation to be fixed.
    @pytest.mark.asyncio
    async def test_hotel_booking_specific_flow(self, travel_agent):
        """
        Test hotel booking specific workflow with targeted flow validation.
        """
        logger.info("🧪 Testing hotel booking specific flow")
        hotel_request = "Book a luxury hotel in Goa for 3 nights starting December 20th"
        result = await travel_agent.process_travel_request(hotel_request)
        
        # Validate response content
        assert "hotel" in result.lower(), f"Response should mention hotel: {result}"
        
        # Validate specific hotel booking interactions using actual LangChain/LangGraph patterns
        hotel_interactions = {
            "participants": self.participants,
            "interactions": [
                ("TC", "HA", "delegation"),   # Coordinator delegates to Hotel Assistant
                ("HA", "BHT", "invocation")    # Hotel Assistant invocation
            ]
        }
        (self.assert_traces()
         .assert_agent_flow(hotel_interactions)
        )
        logger.info("✅ Hotel booking flow validation passed")
    
    @pytest.mark.asyncio
    async def test_recommendations_flow(self, travel_agent):
        """
        Test travel recommendations workflow with flow validation.
        """
        logger.info("🧪 Testing recommendations flow")
        recommendations_request = "What are the best attractions to visit in Delhi?"
        result = await travel_agent.process_travel_request(recommendations_request)
        
        # Validate response content
        assert any(keyword in result.lower() for keyword in ["attraction", "recommend", "delhi"]), \
            f"Response should mention recommendations: {result}"
        
        # Validate specific recommendations interactions using actual LangChain/LangGraph patterns
        recommendations_interactions = {
            "participants": self.participants,
            "interactions": [
                ("TC", "RA", "delegation"),     # Coordinator delegates to Recommendations Assistant
                ("RA", "GRT", "invocation")      # Recommendations Assistant invocation
            ]
        }
        (self.assert_traces()
         .assert_agent_flow(recommendations_interactions)
        )
        logger.info("✅ Recommendations flow validation passed")
    

if __name__ == "__main__":

    logger.info("\n" + "=" * 70)
    logger.info("🧪 Running LangChain Travel Agent Test Suite")
    logger.info("=" * 70)
    
    # Run the test suite
    test_result = pytest.main([
        __file__ + "::TestLangChainTravelAgent",
        "-v", 
        "--tb=short",
        "-s"  # Show print statements
    ])
    
    if test_result == 0:
        logger.info("\n🎉 All LangChain tests passed! The LangChain travel agent is working correctly.")
    else:
        logger.info("\n❌ Some LangChain tests failed. Check the output above.")