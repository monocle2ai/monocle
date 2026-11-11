"""LangGraph travel-booking agent used as the target of scenario harness tests.

Deliberately built to ask for missing details one at a time, so the harness's
progressive-disclosure behaviour has something to exercise.

Uses Azure OpenAI with the same four environment variables as the other live
LangChain tests in this repo (see apptrace/tests/integration/test_langchain_sample.py).
"""
import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

SYSTEM_PROMPT = """You are a travel booking agent. You can book a flight from one \
airport to another for a given date, and a hotel in a given city for a given date.

Never invent missing details. If you do not know the source airport, the destination \
airport or the date, ask the user for exactly one missing detail at a time. Once you \
have everything a booking needs, call the tool and confirm the booking in one sentence."""


@tool
def book_flight(source: str, destination: str, date: str) -> str:
    """Books a flight from a source airport to a destination airport on a given date."""
    return (f"Flight booked from {source} to {destination} on {date}. "
            f"Confirmation code MNCL123.")


@tool
def book_hotel(city: str, date: str) -> str:
    """Books a hotel room in a given city for a given date."""
    return f"Hotel booked in {city} for {date}. Confirmation code MNCL456."


def build_chat_model(temperature: float = 0) -> AzureChatOpenAI:
    """Build an Azure OpenAI chat model from the repo's standard env vars.

    Shared by the target agent and, in the harness test, by the test agent and judge --
    so a single set of credentials drives the whole scenario.
    """
    return AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        temperature=temperature,
    )


def build_travel_agent():
    """Build a fresh travel agent with its own conversation memory."""
    return create_agent(
        model=build_chat_model(),
        tools=[book_flight, book_hotel],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
        name="langgraph_travel_agent",
    )
