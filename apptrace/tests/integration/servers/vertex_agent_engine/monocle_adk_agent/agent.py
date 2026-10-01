"""Flight booking agent for Vertex AI Agent Engine, traced with Monocle.

``adk api_server`` installs Google's tracer provider at startup and imports this
module on the first request, so Monocle, set up here, attaches to that provider.
Configured by MONOCLE_EXPORTER, MONOCLE_WORKFLOW_NAME and the OKAHU_* variables.
"""
import os

from google.adk.agents import Agent
from monocle_apptrace import setup_monocle_telemetry

setup_monocle_telemetry(workflow_name=os.getenv("MONOCLE_WORKFLOW_NAME", "vertex_adk_flight_agent"))


def book_flight(from_airport: str, to_airport: str) -> dict:
    """Books a flight between two airports.

    Args:
        from_airport (str): Departure airport code, e.g. BOM.
        to_airport (str): Destination airport code, e.g. JFK.

    Returns:
        dict: status and booking confirmation.
    """
    return {"status": "success", "confirmation": f"FLIGHT CONFIRMED: {from_airport} to {to_airport}"}


root_agent = Agent(
    name="flight_booking_agent",
    model="gemini-2.5-flash",
    instruction="Book flights with the book_flight tool and report the confirmation.",
    tools=[book_flight],
)
