import os
import logging
import boto3
from strands.agent.agent import Agent
from strands.tools.decorator import tool
from strands.models.bedrock import BedrockModel

logging.basicConfig(level=logging.WARN)

@tool
def strands_book_flight(from_airport: str, to_airport: str, date: str) -> dict:
    """Books a flight from one airport to another.

    Args:
        from_airport (str): The airport from which the flight departs.
        to_airport (str): The airport to which the flight arrives.
        date (str): The date of the flight.

    Returns:
        dict: status and message.
    """
    return {
        "status": "success",
        "message": f"Flight booked from {from_airport} to {to_airport} on {date}."
    }

@tool
def strands_book_hotel(hotel_name: str, city: str, check_in_date: str, duration: int, room_type: str = "standard") -> dict:
    """Books a hotel for a stay.

    Args:
        hotel_name (str): The name of the hotel to book.
        city (str): The city where the hotel is located.
        check_in_date (str): The check-in date for the hotel stay.
        duration (int): The duration of the hotel stay in nights.
        room_type (str): The room type requested, e.g. "two queen" or "king". Defaults to "standard".

    Returns:
        dict: status and message.
    """
    return {
        "status": "success",
        "message": f"Successfully booked a {room_type} room at {hotel_name} in {city} from {check_in_date} for {duration} nights."
    }

# Initialize AWS Bedrock model
boto_session = boto3.Session()
# Set the model explicitly: the Strands default (Claude Sonnet 4) has reached end of life on Bedrock.
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-5-5")
model = BedrockModel(model_id=BEDROCK_MODEL_ID, boto_session=boto_session, streaming=False)

# Create the root agent
root_agent = Agent(
    name="strands_travel_booking_agent",
    model=model,
    system_prompt="""You are a travel booking agent who can assist users in booking flights and hotels.
    You handle both flight and hotel bookings. Provide a consolidated and concise response after completing the bookings.
    Always confirm what you've booked in a single sentence summary.
    Reply with only that one sentence: no follow-up notes, caveats, or suggestions.""",
    tools=[strands_book_flight, strands_book_hotel],
    description="Travel booking agent that handles flight and hotel bookings",
)
