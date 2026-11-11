"""A guardrail stops an agent: told to start every answer with 123456, an ADK agent never gets to answer."""
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
    GUARDRAIL_NAME, GUARDRAIL_REASON, GUARDRAIL_SIDE, MONOCLE_GUARDRAILS_ENV, MONOCLE_INSTRUCTION_ENV,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.status import StatusCode

load_dotenv(find_dotenv(), override=True)

DATA = Path(__file__).parents[1] / "data"
START_WITH_123456_FILE = str(DATA / "start-with-123456.yaml")
STOP_ANSWERS_WITH_123456_FILE = str(DATA / "stop-answers-with-123456.yaml")

flight_booking_agent = LlmAgent(
    name="flight_booking_agent",
    model=LiteLlm(model="gpt-4o-mini"),
    description="Agent to book flights based on user queries.",
    instruction="You are a helpful agent who can assist users in booking flights.",
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


@pytest.fixture(scope="function")
def setup():
    instrumentor = None
    os.environ[MONOCLE_INSTRUCTION_ENV] = START_WITH_123456_FILE
    os.environ[MONOCLE_GUARDRAILS_ENV] = STOP_ANSWERS_WITH_123456_FILE
    try:
        memory_exporter = InMemorySpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="adk_guardrail_stops_agent",
            span_processors=[SimpleSpanProcessor(memory_exporter)],
        )
        yield memory_exporter
    finally:
        del os.environ[MONOCLE_INSTRUCTION_ENV]
        del os.environ[MONOCLE_GUARDRAILS_ENV]
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_agent_is_stopped_before_it_can_give_the_answer_the_guardrail_forbids(setup):
    with pytest.raises(Exception, match="Kill switch triggered by guardrail 'stop-answers-with-123456' on the output"):
        await final_answer("Which airports are in New York?")

    stopped_spans = [span for span in setup.get_finished_spans() if GUARDRAIL_NAME in span.attributes]
    assert stopped_spans
    for span in stopped_spans:
        assert span.attributes["span.type"].startswith("inference")
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes[GUARDRAIL_NAME] == "stop-answers-with-123456"
        assert span.attributes[GUARDRAIL_SIDE] == "output"
        assert span.attributes[GUARDRAIL_REASON] == "pattern 1"
