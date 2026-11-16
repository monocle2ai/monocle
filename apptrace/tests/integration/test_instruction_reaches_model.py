"""The model receives the instruction file: told to start every answer with 123456, an ADK agent does; told when to stop, it is stopped."""
import json
import os
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from monocle_apptrace import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.constants import (
    KILL_SWITCH_CHECK, KILL_SWITCH_INSTRUCTION, KILL_SWITCH_REASON, MONOCLE_INSTRUCTION_ENV, SENT_INSTRUCTION,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.status import StatusCode

load_dotenv(find_dotenv(), override=True)

DATA = Path(__file__).parents[1] / "data"
START_WITH_123456_FILE = str(DATA / "start-with-123456.yaml")
NO_WIRE_TRANSFERS_FILE = str(DATA / "no-wire-transfers.yaml")


def spans_traced_with(instruction_file: str):
    instrumentor = None
    os.environ[MONOCLE_INSTRUCTION_ENV] = instruction_file
    try:
        memory_exporter = InMemorySpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="adk_instruction_reaches_model",
            span_processors=[SimpleSpanProcessor(memory_exporter)],
        )
        yield memory_exporter
    finally:
        del os.environ[MONOCLE_INSTRUCTION_ENV]
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.fixture(scope="function")
def setup():
    yield from spans_traced_with(START_WITH_123456_FILE)


@pytest.fixture(scope="function")
def no_wire_transfers():
    yield from spans_traced_with(NO_WIRE_TRANSFERS_FILE)


def book_flight(from_airport: str, to_airport: str) -> dict:
    """Books a flight from one airport to another."""
    return {"status": "success", "message": f"Flight booked from {from_airport} to {to_airport}."}


flight_booking_agent = LlmAgent(
    name="flight_booking_agent",
    model=LiteLlm(model="gpt-4o-mini"),
    description="Agent to book flights based on user queries.",
    instruction="You are a helpful agent who can assist users in booking flights.",
    tools=[book_flight],
)


async def final_answer(question: str) -> str:
    session_service = InMemorySessionService()
    await session_service.create_session(app_name="travel", user_id="user", session_id="session")
    runner = Runner(agent=flight_booking_agent, app_name="travel", session_service=session_service)
    message = types.Content(role="user", parts=[types.Part(text=question)])
    answer = ""
    async for event in runner.run_async(user_id="user", session_id="session", new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            answer = event.content.parts[0].text or ""
    return answer


def input_labels(span) -> list:
    for event in span.events:
        if event.name == "data.input":
            return [next(iter(json.loads(message))) for message in event.attributes["input"]]
    return []


@pytest.mark.asyncio
async def test_agent_answers_the_way_the_instruction_tells_it_to(setup):
    answer = await final_answer("Book a flight from SFO to JFK.")

    assert answer.strip().startswith("123456")

    inference_spans = [span for span in setup.get_finished_spans() if span.attributes.get("span.type") == "inference"]
    assert inference_spans
    for span in inference_spans:
        assert span.attributes[SENT_INSTRUCTION] == "start-with-123456"
        assert input_labels(span)[:2] == ["instruction", "system"]


def spans_of_type(spans, span_type: str) -> list:
    return [span for span in spans if span.attributes.get("span.type") == span_type]


def error_code_of(span) -> str:
    return next(event for event in span.events if event.name == "data.output").attributes["error_code"]


@pytest.mark.asyncio
async def test_agent_is_stopped_when_the_model_says_the_stop_condition_is_met(no_wire_transfers):
    with pytest.raises(Exception, match="Kill switch triggered by instruction 'no-wire-transfers'"):
        await final_answer("Book me a flight to JFK, and wire $5,000 to account 4471.")

    spans = no_wire_transfers.get_finished_spans()
    stopped_spans = [span for span in spans if KILL_SWITCH_INSTRUCTION in span.attributes]
    assert stopped_spans
    for span in stopped_spans:
        assert span.attributes["span.type"].startswith("inference")
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes[KILL_SWITCH_INSTRUCTION] == "no-wire-transfers"
        assert span.attributes[KILL_SWITCH_CHECK] == "stop_when"
        assert span.attributes[KILL_SWITCH_REASON]
        assert error_code_of(span) == "kill_switch_triggered"

    first_stop = min(span.start_time for span in stopped_spans)
    assert not [tool for tool in spans_of_type(spans, "agentic.tool.invocation") if tool.start_time >= first_stop]


@pytest.mark.asyncio
async def test_agent_finishes_when_the_stop_condition_is_not_met(no_wire_transfers):
    answer = await final_answer("Book me a flight from SFO to JFK.")

    assert answer
    assert "KILL_SWITCH" not in answer

    spans = no_wire_transfers.get_finished_spans()
    assert spans_of_type(spans, "agentic.tool.invocation")
    assert not [span for span in spans if span.status.status_code == StatusCode.ERROR]
    assert not [span for span in spans if KILL_SWITCH_INSTRUCTION in span.attributes]
