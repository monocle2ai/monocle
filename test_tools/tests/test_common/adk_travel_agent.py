import os
import pytest 
import logging

from monocle_test_tools import TestCase, MonocleValidator
logging.basicConfig(level=logging.WARN)

import os
import time
from google.adk.agents import LlmAgent, SequentialAgent

MAX_OUTPUT_TOKENS = 50

def adk_book_flight_5(from_airport: str, to_airport: str) -> dict:
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
        "message": f"Flight booked from {from_airport} to {to_airport}."
    }

def adk_book_hotel_5(hotel_name: str, city: str, check_in_date: str, duration: int) -> dict:
    """Books a hotel for a stay.

    Args:
        hotel_name (str): The name of the hotel to book.
        city (str): The city where the hotel is located.
        check_in_date (str): The check-in date for the hotel stay.
        duration (int): The duration of the hotel stay in nights.

    Returns:
        dict: status and message.
    """
    return {
        "status": "success",
        "message": f"Successfully booked a stay at {hotel_name} in {city} from {check_in_date} for {duration} nights."
    }

flight_booking_agent = LlmAgent(
    name="adk_flight_booking_agent_5",
    model="gemini-2.0-flash",
    description= "Agent to book flights based on user queries.",
    instruction= "You are a helpful agent who can assist users in booking flights. You only handle flight booking. Just handle that part from what the user says, ignore other parts of the requests.",
    tools=[adk_book_flight_5] 
)

hotel_booking_agent = LlmAgent(
    name="adk_hotel_booking_agent_5",
    model="gemini-2.0-flash",
    description= "Agent to book hotels based on user queries.",
    instruction= "You are a helpful agent who can assist users in booking hotels. You only handle hotel booking. Just handle that part from what the user says, ignore other parts of the requests.",
    tools=[adk_book_hotel_5] 
)

trip_summary_agent = LlmAgent(
    name="adk_trip_summary_agent_5",
    model="gemini-2.0-flash",
    description= "Summarize the travel details from hotel bookings and flight bookings agents.",
    instruction= "Summarize the travel details from hotel bookings and flight bookings agents. Be concise in response and provide a single sentence summary.",
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
