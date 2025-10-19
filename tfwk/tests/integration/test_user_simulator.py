
import textwrap

import pytest
from agentx.langchain_travel_agent import LangChainTravelAgentDemo
from monocle_tfwk.agents.simulator import TestAgent


@pytest.mark.asyncio
async def test_travel_agent_vague_user():
    """Test the travel agent with a vague user persona."""
    booking_app = LangChainTravelAgentDemo()
    test_persona_1 = {
        "goal": "Book a flight to Paris for 2 people tomorrow.",
        "data": {
            # "source": "New York",
            "destination": "Paris",
            "date": "tomorrow",
            "passengers": 2
        },
        "initial_query_prompt": textwrap.dedent("""
            You are a user simulator starting a conversation with a booking agent.
            Your goal is: {goal}
            Your data is: {persona_data}
            Start the conversation VAGUELY. Do not provide all the information. 
            Just say something like "Hi, I need to book a flight."
        """).strip(),
        "clarification_prompt": textwrap.dedent("""
            You are a user simulator answering a clarification question from a booking agent.
            Your goal is to provide answers to complete the clarification.
            You use the following persona data:
            {persona_data}

            You will be given the conversation history and a final question from the agent.
            Your job is to answer ONLY that final question, using the information from your persona data,
            or make up any missing details.
            - Do not be overly conversational.
            - Just provide the specific information requested.
        """).strip()
    }

    tester = TestAgent(tool_to_test=booking_app.process_travel_request, user_persona=test_persona_1)
    history = await tester.run_test()
    # Basic assertion: history should not be empty
    assert history, "Conversation history should not be empty"
    # Optionally, check that the goal destination is mentioned in the conversation
    assert any("Paris" in str(turn) for turn in history), "Destination 'Paris' should appear in conversation"

if __name__ == "__main__":
    # Run the test directly if executed as a script
    pytest.main([__file__, "-s", "--tb=short"])