import os
import sys
from typing import Annotated
import random

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIAssistantsClient
from azure.identity import DefaultAzureCredential
from azure.ai.agentserver.agentframework import from_agent_framework

from dotenv import load_dotenv
load_dotenv()

# Configure these for Azure OpenAI Assistants mode
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_DEPLOYMENT = os.getenv("AZURE_OPENAI_API_DEPLOYMENT")

# Flight booking tool
def book_flight(
    from_airport: Annotated[str, "The EXACT departure location extracted from the user's current message - can be airport code or city name"],
    to_airport: Annotated[str, "The EXACT destination location extracted from the user's current message - can be airport code or city name"],
) -> str:
    """Book a flight between two locations. Extract the departure and destination from the user's message."""
    confirmation = f"FL{random.randint(100000, 999999)}"
    cost = random.randint(300, 800)
    result = f"FLIGHT BOOKING CONFIRMED #{confirmation}: {from_airport} to {to_airport} - ${cost}"

    return result

def book_hotel(
    hotel_name: Annotated[str, "The EXACT hotel name extracted from the user's current message"],
    city: Annotated[str, "The EXACT city extracted from the user's current message"]) -> str:
    """Books a hotel for a stay.

    Args:
        hotel_name (str): The name of the hotel to book.
        city (str): The city where the hotel is located.

    Returns:
        dict: status and message.
    """
    return f"Successfully booked a stay at {hotel_name} in {city}."

def get_agent_for_hosted_mode():
    """Create agent for Azure Hosted Agent deployment"""
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_DEPLOYMENT:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_DEPLOYMENT must be set in .env file!"
        )

    chat_client = AzureOpenAIAssistantsClient(
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_API_DEPLOYMENT,
        api_version="2024-05-01-preview",
        credential=DefaultAzureCredential(),
    )

    return ChatAgent(
        name="MS Flight Agent",
        chat_client=chat_client,
        instructions="""You are a flight booking assistant.

Read the user's current message and extract:
- Departure location (the place they want to fly FROM)
- Destination location (the place they want to fly TO)

Then call the book_flight function with those exact locations from their message.""",
        tools=[book_flight, book_hotel],
    )

def start_msgent_server():    
    agent = get_agent_for_hosted_mode()
    from_agent_framework(agent).run()