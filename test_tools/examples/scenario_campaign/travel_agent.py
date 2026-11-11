"""A small LangGraph travel agent to run the example campaign against.

Books flights and hotels with mock tools, and is told to ask for one missing detail at
a time, so the withheld-param scenarios have something to exercise. Uses OpenAI
through OPENAI_API_KEY; set OPENAI_MODEL to change the model (default gpt-4.1-mini).
"""
import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

SYSTEM_PROMPT = """You are a travel booking agent. You can book a flight from one \
airport to another for a given date, and a hotel in a given city for a given date.

Never invent missing details. If you do not know the source airport, the destination \
airport or the date, ask the user for exactly one missing detail at a time. Once you \
have everything a booking needs, call the tool and confirm the booking in one sentence. \
You cannot help with refunds, visas or anything other than booking."""


@tool
def book_flight(source: str, destination: str, date: str) -> str:
    """Books a flight from a source airport to a destination airport on a given date."""
    return f"Flight booked from {source} to {destination} on {date}. Confirmation code MNCL123."


@tool
def book_hotel(city: str, date: str) -> str:
    """Books a hotel room in a given city for a given date."""
    return f"Hotel booked in {city} for {date}. Confirmation code MNCL456."


def build_chat_model(temperature: float = 0) -> ChatOpenAI:
    """One OpenAI model for the agent, the tester and the judge."""
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=temperature)


def build_travel_agent():
    """A fresh agent with its own conversation memory."""
    return create_agent(model=build_chat_model(), tools=[book_flight, book_hotel],
                        system_prompt=SYSTEM_PROMPT, checkpointer=MemorySaver(),
                        name="example_travel_agent")
