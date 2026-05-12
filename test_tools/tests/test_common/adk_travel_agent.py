import asyncio
import logging
import os
import uuid

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"  # Set to TRUE to use Vertex AI
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))
# Set env model as gemini-2.5-flash-lite by default
GOOGLE_GENAI_MODEL = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash-lite")

# ---------------------------------------------------------------------------
# Legacy "_5" suffixed tools/agents kept for back-compat with existing tests
# (test_adk_travel_agent.py, test_adk_custom_assertions.py, test_evals.py,
#  test_tool_mock.py, test_performance.py, test_adk_travel_agent_fluent.py).
# These exports are required: `root_agent` and `root_agent_parallel`.
# ---------------------------------------------------------------------------

def adk_book_flight_5(from_airport: str, to_airport: str) -> dict:
    """Books a flight from one airport to another."""
    return {
        "status": "success",
        "message": f"Flight booked from {from_airport} to {to_airport}."
    }

def adk_book_hotel_5(hotel_name: str, city: str, check_in_date: str, duration: int) -> dict:
    """Books a hotel for a stay."""
    return {
        "status": "success",
        "message": f"Successfully booked a stay at {hotel_name} in {city} from {check_in_date} for {duration} nights."
    }

flight_booking_agent = LlmAgent(
    name="adk_flight_booking_agent_5",
    model="gemini-2.0-flash",
    description="Agent to book flights based on user queries.",
    instruction="You are a helpful agent who can assist users in booking flights. You only handle flight booking. Just handle that part from what the user says, ignore other parts of the requests.",
    tools=[adk_book_flight_5]
)

hotel_booking_agent = LlmAgent(
    name="adk_hotel_booking_agent_5",
    model="gemini-2.0-flash",
    description="Agent to book hotels based on user queries.",
    instruction="You are a helpful agent who can assist users in booking hotels. You only handle hotel booking. Just handle that part from what the user says, ignore other parts of the requests.",
    tools=[adk_book_hotel_5]
)

trip_summary_agent = LlmAgent(
    name="adk_trip_summary_agent_5",
    model="gemini-2.0-flash",
    description="Summarize the travel details from hotel bookings and flight bookings agents.",
    instruction="Summarize the travel details from hotel bookings and flight bookings agents. Be concise in response and provide a single sentence summary.",
    output_key="booking_summary"
)

root_agent = SequentialAgent(
    name="adk_supervisor_agent_5",
    description=
        """
            You are the supervisor agent that coordinates the flight booking and hotel booking.
            You must provide a consolidated summary back to the full coordination of the user's request.
        """
    ,
    sub_agents=[flight_booking_agent, hotel_booking_agent, trip_summary_agent],
)

# Separate agent instances are required for the ParallelAgent because ADK
# does not allow re-parenting the same LlmAgent into a second tree.
flight_booking_agent_parallel = LlmAgent(
    name="adk_flight_booking_agent_5",
    model="gemini-2.0-flash",
    description="Agent to book flights based on user queries.",
    instruction="You are a helpful agent who can assist users in booking flights. You only handle flight booking. Just handle that part from what the user says, ignore other parts of the requests.",
    tools=[adk_book_flight_5]
)

hotel_booking_agent_parallel = LlmAgent(
    name="adk_hotel_booking_agent_5",
    model="gemini-2.0-flash",
    description="Agent to book hotels based on user queries.",
    instruction="You are a helpful agent who can assist users in booking hotels. You only handle hotel booking. Just handle that part from what the user says, ignore other parts of the requests.",
    tools=[adk_book_hotel_5]
)

trip_summary_agent_parallel = LlmAgent(
    name="adk_trip_summary_agent_5",
    model="gemini-2.0-flash",
    description="Summarize the travel details from hotel bookings and flight bookings agents.",
    instruction="Summarize the travel details from hotel bookings and flight bookings agents. Be concise in response and provide a single sentence summary.",
    output_key="booking_summary"
)

parallel_booking_agent = ParallelAgent(
    name="adk_parallel_booking_coordinator_5",
    description="Coordinates flight and hotel booking in parallel",
    sub_agents=[flight_booking_agent_parallel, hotel_booking_agent_parallel],
)

root_agent_parallel = SequentialAgent(
    name="adk_supervisor_agent_parallel_5",
    description=
        """
            You are the supervisor agent that coordinates parallel flight and hotel booking,
            then provides a consolidated summary.
        """
    ,
    sub_agents=[parallel_booking_agent, trip_summary_agent_parallel],
)

# ---------------------------------------------------------------------------
# New non-suffixed agents used by the FastAPI remote-agent test path.
# These power `runner` / `run_agent()` and are imported by remote_agent_test.py
# as `remote_root_agent`.
# ---------------------------------------------------------------------------

def adk_book_flight(from_airport: str, to_airport: str) -> dict:
    """Books a flight from one airport to another.

    Args:
        from_airport (str): The airport from which the flight departs.
        to_airport (str): The airport to which the flight arrives.

    Returns:
        dict: status and message.
    """
    return {
        "status": "success",
        "message": f"Flight booked from {from_airport} to {to_airport}."
    }

def adk_book_hotel(hotel_name: str, city: str) -> dict:
    """Books a hotel for a stay.

    Args:
        hotel_name (str): The name of the hotel to book.
        city (str): The city where the hotel is located.

    Returns:
        dict: status and message.
    """
    return {
        "status": "success",
        "message": f"Successfully booked a stay at {hotel_name} in {city}."
    }

contentConfig: types.GenerateContentConfig = types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS)

remote_flight_booking_agent = LlmAgent(
    name="adk_flight_booking_agent",
    model=GOOGLE_GENAI_MODEL,
    description="Agent to book flights based on user queries.",
    instruction="You are a helpful agent who can assist users in booking flights. You only handle flight booking. Just handle that part from what the user says, ignore other parts of the requests.",
    generate_content_config=contentConfig,
    tools=[adk_book_flight]
)

remote_hotel_booking_agent = LlmAgent(
    name="adk_hotel_booking_agent",
    model=GOOGLE_GENAI_MODEL,
    description="Agent to book hotels based on user queries.",
    instruction="You are a helpful agent who can assist users in booking hotels. You only handle hotel booking. Book hotel if the user explicitly asks, just handle that part from what the user says, ignore other parts of the requests. NOTE: Marriott is only available on odd dates. Otherwise Hilton is the primary option unless user states specific hotel criteria and you can go ahead and book that instead.",
    generate_content_config=contentConfig,
    tools=[adk_book_hotel]
)

remote_trip_summary_agent = LlmAgent(
    name="adk_trip_summary_agent",
    model=GOOGLE_GENAI_MODEL,
    description="Summarize the travel details from hotel bookings and flight bookings agents.",
    instruction="Summarize the travel details from hotel bookings and flight bookings agents. Be concise in response and provide a single sentence summary.",
    generate_content_config=contentConfig,
    output_key="booking_summary"
)

remote_root_agent = SequentialAgent(
    name="adk_supervisor_agent",
    description=
        """
            You are the supervisor agent that coordinates the flight booking and hotel booking.
            You must provide a consolidated summary back to the full coordination of the user's request.
        """
    ,
    sub_agents=[remote_flight_booking_agent, remote_hotel_booking_agent, remote_trip_summary_agent],
)

session_service = InMemorySessionService()
APP_NAME = "streaming_app"
USER_ID = "user_123"
SESSION_ID = "session_456"

runner = Runner(
    agent=remote_root_agent,
    app_name=APP_NAME,
    session_service=session_service
)

async def run_agent(test_message: str):
    session_id = str(uuid.uuid4())
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id
    )
    content = types.Content(role='user', parts=[types.Part(text=test_message)])
    response = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content
    ):
        if event.is_final_response():
            response = event.content

    return response.parts[0].text

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    user_request = input("\nI am a travel booking agent. How can I assist you with your travel plans? ")
    asyncio.run(run_agent(user_request))
