import datetime
import logging
import os
import time
from zoneinfo import ZoneInfo

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from config.conftest import temporary_env_var
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from monocle_apptrace import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup():
    try:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        custom_exporter = CustomConsoleSpanExporter()
        memory_exporter = InMemorySpanExporter()
        span_processors = [SimpleSpanProcessor(memory_exporter), SimpleSpanProcessor(custom_exporter)]
        instrumentor = setup_monocle_telemetry(
            workflow_name="langchain_agent_1",
            span_processors=span_processors
        )
        yield memory_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city."
    ),
    tools=[get_weather, get_current_time],
)

session_service = InMemorySessionService()
APP_NAME = "streaming_app"
USER_ID = "user_123"
SESSION_ID = "session_456"



runner = Runner(
    agent=root_agent,  # Assume this is defined
    app_name=APP_NAME,
    session_service=session_service
)

async def run_agent(test_message: str):
    session = await session_service.create_session(
        app_name=APP_NAME, 
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    content = types.Content(role='user', parts=[types.Part(text=test_message)])
    # Process events as they arrive using async for
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        # For final response
        if event.is_final_response():
            logger.info(event.content)  # End line after response

@pytest.mark.asyncio
async def test_multi_agent(setup):
    test_message = "What is the current weather in New York?"
    await run_agent(test_message)
    verify_spans(setup)

@pytest.mark.asyncio
async def test_invalid_api_key_error_code_in_span(setup):
    """Test that passing an invalid API key results in error_code in the span."""
    # Simulate invalid API key by setting an obviously wrong value
    with temporary_env_var("GOOGLE_API_KEY", "INVALID_API_KEY"):
        try:
            test_message = "What is the current weather in New York?"
            await run_agent(test_message)
            time.sleep(2)
        except Exception:
            spans = setup.get_finished_spans()
            for span in spans:
                span_attributes = span.attributes
                if (
                        "span.type" in span_attributes
                        and span_attributes["span.type"] == "agentic.invocation"
                        and "entity.1.name" in span_attributes
                ):
                    span_input, span_output, _ = span.events
                    assert "error_code" in span_output.attributes
                    assert span_output.attributes["error_code"] == 400

def verify_spans(memory_exporter):
    time.sleep(2)
    found_inference = found_agent = found_tool = False
    spans = memory_exporter.get_finished_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
                span_attributes["span.type"] == "inference"
                or span_attributes["span.type"] == "inference.framework"
        ):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.gemini"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gemini-2.0-flash"
            assert span_attributes["entity.2.type"] == "model.llm.gemini-2.0-flash"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
            found_inference = True

        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.invocation"
                and "entity.1.name" in span_attributes
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.adk"
            if span_attributes["entity.1.name"] == "weather_time_agent":
                found_agent = True
        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.adk"
            if span_attributes["entity.1.name"] == "get_weather":
                found_tool = True

        if 'monocle_apptrace.version' in span_attributes:
            assert "scope.agentic.session" in span_attributes, f"scope.agentic.session not found in span {span.name}"
            assert span_attributes["scope.agentic.session"] == SESSION_ID, f"Expected session {SESSION_ID}, got {span_attributes.get('scope.agentic.session')}"

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
