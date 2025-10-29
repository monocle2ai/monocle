import json
import logging
import textwrap

import pytest
from agentx.langchain_travel_agent import LangChainTravelAgentDemo
from monocle_tfwk import BaseAgentTest
from monocle_tfwk.agents.simulator import TestAgent
from monocle_tfwk.schema import MonocleSpanType
from monocle_tfwk.visualization.gantt_chart import VisualizationMode

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
            "forbidden": [("U", "FA"), ("U", "HA"), ("U", "RA")],  # Direct interactions not allowed
            "interactions": [
                # Match actual LangChain/LangGraph span structure using MonocleSpanType enum values
                #("U", "TC", MonocleSpanType.AGENTIC_REQUEST),  # User invokes Travel Coordinator
                # ("TC", "RA", MonocleSpanType.AGENTIC_DELEGATION),         # TC delegates to RA
                # ("RA", "GRT", MonocleSpanType.AGENTIC_TOOL_INVOCATION),         # Recommendations Assistant using get_recommendations_tool
                # ("TC", "FA", MonocleSpanType.AGENTIC_DELEGATION),         # TC delegates to FA (agentic.delegation)
                ("FA", "BFT", MonocleSpanType.AGENTIC_TOOL_INVOCATION),         # Flight Assistant using book_flight_tool
                # ("TC", "HA", MonocleSpanType.AGENTIC_DELEGATION),          # TC delegates to HA  
                ("HA", "BHT", MonocleSpanType.AGENTIC_TOOL_INVOCATION),         # Hotel Assistant using book_hotel_tool
            ]
        }

        logger.info("🧪 Testing hotel booking specific flow")
        
        test_persona = {
            "goal": "Recommend places to see in Mumbai and book business class flight from Delhi to Mumbai on December 15th, 2025 for a single traveller and luxury hotel for 2 nights starting December 15th.",
            "persona_data": {
                "destination": "Mumbai",
                "date": "December 15th, 2025",
                "passengers": 1,
                "hotel_nights": 2
            },
            "initial_query_prompt": textwrap.dedent("""
                You are a user conversing with a travel agent.
                Your goal is: {goal}
                Your data is: {persona_data}
                Just say something like "Hi, plan my stay and travel to Mumbai"
            """).strip()
        }

        # Set up the second TestAgent, passing the *method* as the tool
        try:
            tester = TestAgent(tool_to_test=travel_agent.process_travel_request, user_persona=test_persona)
            # Run the simulation (now awaited)
            conversation = await tester.run_test(15)
            assert any(keyword in str(conversation[-1]).lower() for keyword in ["flight", "hotel", "mumbai"]), \
                f"Response should mention travel elements: {conversation}"

        except (ValueError, TypeError) as e:
            assert False, f"❌ failed to create user simulator: {e}"
        
        
        # === SIMPLIFIED AGENTIC FLOW VALIDATION BASED ON OBSERVED PATTERNS ===
        logger.info("=== Gantt Chart Visualization ===")
        self.display_flow_gantt_chart(VisualizationMode.DETAILED)
        # Assert the realistic flow pattern
        traces = self.assert_traces()
        (traces  # Get trace assertions
         .assert_agent_flow(whole_flow)
        )

        # Test the JSON format confirmation
        confirmation = await traces.ask_llm_about_traces(
            "Is the flight and hotel booking confirmed? "
            "Return your answer in JSON format with this exact structure: "
            '{"confirmed": true/false, "reason": "brief explanation"}'
        )
        
        # Extract and assert the confirmed field
  
        confirmation_data = json.loads(confirmation)
        assert confirmation_data["confirmed"], f"Hotel booking should be confirmed: {confirmation}"
        logger.info(f"✅ Full travel booking flow validation passed {confirmation}")
        


    @pytest.mark.asyncio
    async def test_flight_booking_specific_flow(self, travel_agent):
        """
        Test flight booking specific workflow with targeted flow validation.
        """
        logger.info("🧪 Testing flight booking specific flow")
        flight_request = "Book a business class flight from Delhi to Mumbai on December 15th, 2025 without waiting for my response"
        result = await travel_agent.process_travel_request(flight_request)
        
        # Validate response content
        assert "flight" in result.lower(), f"Response should mention flight: {result}"
        
        # Validate specific flight booking interactions using actual LangChain/LangGraph patterns
        flight_interactions = {
            "participants": self.participants,
            "interactions": [
                #("TC", "FA", MonocleSpanType.AGENTIC_DELEGATION),  # Coordinator delegates to Flight Assistant
                ("FA", "BFT", MonocleSpanType.AGENTIC_TOOL_INVOCATION)   # Flight Assistant invocation
            ]
        }
        (self.assert_traces()
         .assert_agent_flow(flight_interactions)  # Legacy tuple format still supported
        )
        logger.info("✅ Flight booking flow validation passed")

    

    @pytest.mark.asyncio
    async def test_hotel_booking_specific_flow(self, travel_agent):
        """
        Test hotel booking specific workflow with targeted flow validation.
        """
        logger.info("🧪 Testing hotel booking specific flow")
        
        test_persona = {
            "goal": "Book a luxury hotel in Goa for 3 nights starting December 20th",
            "persona_data": {
                "destination": "Goa",
                "date": "December 20th, 2025",
                "passengers": 4
            },
            "initial_query_prompt": textwrap.dedent("""
                You are a user simulator starting a conversation with a booking agent.
                Your goal is: {goal}
                Your data is: {persona_data}
                Start the conversation VAGUELY. Do not provide all the information. 
                Just say something like "Hi, I need to book a hotel."
            """).strip()
        }

        # Set up the second TestAgent, passing the *method* as the tool
        try:
            user_simulator = TestAgent(tool_to_test=travel_agent.process_travel_request, user_persona=test_persona)
            conversation = await user_simulator.run_test()
            assert any(keyword in str(conversation[-1]).lower() for keyword in ["hotel", "goa"]), \
                f"Response should mention travel elements: {conversation}"
        except (ValueError, TypeError) as e:
             assert False, f"❌ failed to create user simulator: {e}"

        
        # logger.info("=== Gantt Chart Visualization ===")
        # self.display_flow_gantt_chart(VisualizationMode.DETAILED)

        # Validate specific hotel booking interactions using actual LangChain/LangGraph patterns
        hotel_interactions = {
            "participants": self.participants,
            "interactions": [
                #("TC", "HA", MonocleSpanType.AGENTIC_DELEGATION),   # Coordinator delegates to Hotel Assistant
                ("HA", "BHT", MonocleSpanType.AGENTIC_TOOL_INVOCATION)    # Hotel Assistant invocation
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
                #("TC", "RA", MonocleSpanType.AGENTIC_DELEGATION),     # Coordinator delegates to Recommendations Assistant
                ("RA", "GRT", MonocleSpanType.AGENTIC_TOOL_INVOCATION)      # Recommendations Assistant invocation
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