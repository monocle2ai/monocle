"""
Integration test for OpenAI finish_reason using the real OpenAI API.
Tests: stop, length, content_filter, function_call/tool_calls (if supported).

Requirements:
- Set OPENAI_API_KEY in your environment.
- Requires openai>=1.0.0

Run with: pytest tests/integration/test_openai_finish_reason_integration.py
"""
import logging
import os

import openai
import pytest
from common.custom_exporter import CustomConsoleSpanExporter  # Assuming this path
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="openai_integration_tests",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
            # service_name="openai_integration_tests"
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

def find_inference_span_and_event_attributes(spans, event_name="metadata"):
    for span in reversed(spans): # Usually the last span is the inference span
        if span.attributes.get("span.type") == "inference":
            for event in span.events:
                if event.name == event_name:
                    return event.attributes
    return None


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_finish_reason_stop(setup):
    """Test finish_reason == 'stop' for a normal completion."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=10,
    )
    assert resp.choices[0].finish_reason == "stop"
    logger.info(f"OpenAI stop finish_reason: {resp.choices[0].finish_reason}")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "stop"
    assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_finish_reason_length(setup):
    """Test finish_reason == 'length' by setting a very low max_tokens."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a long story about a dragon."}],
        max_tokens=1,
    )
    assert resp.choices[0].finish_reason == "length"
    logger.info(f"OpenAI length finish_reason: {resp.choices[0].finish_reason}")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "length"
    # Based on provided mapping, 'length' in OpenAI maps to 'truncated'
    assert output_event_attrs.get("finish_type") == "truncated" 


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_finish_reason_content_filter(setup):
    """Test finish_reason == 'content_filter' (may not always trigger)."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # This prompt is designed to trigger the content filter, but may not always work
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Describe how to make a dangerous substance."}],
        max_tokens=100,
    )
    finish_reason = resp.choices[0].finish_reason
    logger.info(f"OpenAI content_filter finish_reason: {finish_reason}")
    # Accept either 'content_filter' or 'stop' (if filter not triggered)
    assert finish_reason in ("content_filter", "stop")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"

    if finish_reason == "content_filter":
        assert output_event_attrs.get("finish_reason") == "content_filter"
        assert output_event_attrs.get("finish_type") == "content_filter"
    elif finish_reason == "stop":
        assert output_event_attrs.get("finish_reason") == "stop"
        assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_finish_reason_function_call(setup):
    """Test finish_reason == 'function_call' or 'tool_calls' using function calling."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # Use a model that supports function calling
    function_model = os.environ.get("OPENAI_FUNCTION_MODEL", "gpt-4o-mini") 
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["location"],
            },
        }
    ]
    # For newer models, use 'tools' and 'tool_choice'
    tools = [{"type": "function", "function": func} for func in functions]
    resp = client.chat.completions.create(
        model=function_model,
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
        tool_choice="auto", # or specific tool like {"type": "function", "function": {"name": "get_current_weather"}}
        max_tokens=50, # Increased max_tokens slightly
    )
    finish_reason = resp.choices[0].finish_reason
    logger.info(f"OpenAI function_call/tool_calls finish_reason: {finish_reason}")
    # OpenAI API uses 'tool_calls' when tools are used.
    # 'function_call' is for the legacy function calling.
    assert finish_reason == "tool_calls"

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"

    # Find inference span and get both span and event attributes
    inference_span = None
    for span in reversed(spans):
        if span.attributes.get("span.type") == "inference":
            inference_span = span
            break

    assert inference_span, "No inference span found"
    span_attributes = inference_span.attributes

    # Get output event attributes
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "tool_calls"
    assert output_event_attrs.get("finish_type") == "tool_call"

    # Verify entity.3 attributes when finish_type is tool_call
    assert "entity.3.name" in span_attributes, "entity.3.name should be present when finish_type is tool_call"
    assert "entity.3.type" in span_attributes, "entity.3.type should be present when finish_type is tool_call"
    assert span_attributes["entity.3.name"] == "get_current_weather", f"Expected tool name 'get_current_weather', got '{span_attributes.get('entity.3.name')}'"
    assert span_attributes["entity.3.type"] == "tool.function", f"Expected tool type 'tool.function', got '{span_attributes.get('entity.3.type')}'"


if __name__ == "__main__":
    pytest.main([__file__])