import os
import time
from google.adk.agents import LlmAgent, SequentialAgent

def book_flight(from_airport: str, to_airport: str) -> dict:
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

def book_hotel(hotel_name: str, city: str) -> dict:
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

flight_booking_agent = LlmAgent(
    name="flight_assistant",
    model="gemini-2.0-flash",
    description=(
        "Agent to book flights based on user queries."
    ),
    instruction=(
        "You are a helpful agent who can assist users in booking flights."
    ),
    tools=[book_flight]  # Define flight booking tools here
)

hotel_booking_agent = LlmAgent(
    name="hotel_assistant",
    model="gemini-2.0-flash",
    description=(
        "Agent to book hotels based on user queries."
    ),
    instruction=(
        "You are a helpful agent who can assist users in booking hotels."
    ),
    tools=[book_hotel]  # Define hotel booking tools here
)

root_agent = SequentialAgent(
    name="supervisor",
    description=(
        "Supervisor agent that coordinates the flight booking and hotel booking. Provide a consolidated response."
    ),
    sub_agents=[flight_booking_agent, hotel_booking_agent],
)


