import logging
import os

import pytest
from mistralai import Mistral
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small")

@pytest.fixture(scope="module")
def setup():
    try:
        # Setup telemetry
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="mistral_integration_tests",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def find_inference_span_and_event_attributes(spans, event_name="metadata"):
    for span in reversed(spans): # Usually the last span is the inference span
        if span.attributes.get("span.type") == "inference":
            for event in span.events:
                if event.name == event_name:
                    return event.attributes
    return None

@pytest.mark.skipif(not MISTRAL_API_KEY, reason="MISTRAL_API_KEY not set")
def test_finish_reason_stop(setup):
    """Test finish_reason == 'stop' for a normal completion."""
    client = Mistral(api_key=MISTRAL_API_KEY)
    resp = client.chat.complete(
        model=MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "Say hello."}],
    )
    assert resp.choices[0].finish_reason == "stop"
    logger.info("stop finish_reason: %s", resp.choices[0].finish_reason)

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "stop"
    assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not MISTRAL_API_KEY, reason="MISTRAL_API_KEY not set")
def test_finish_reason_length(setup):
    """Test finish_reason == 'length' by setting a very low max_tokens."""
    client = Mistral(api_key=MISTRAL_API_KEY)
    resp = client.chat.complete(
        model=MODEL,
        max_tokens=1,
        messages=[{"role": "user", "content": "Tell me a long story about a dragon."}],
    )
    assert resp.choices[0].finish_reason == "length"
    logger.info("length finish_reason: %s", resp.choices[0].finish_reason)

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "length"
    assert output_event_attrs.get("finish_type") == "truncated"

def find_inference_span_with_tool_call(spans):
    """Find inference span that contains tool calling information."""
    for span in reversed(spans):
        span_type = span.attributes.get("span.type")
        if span_type in ("inference.framework", "inference"):
            # Check if this span has entity.3 attributes (tool info)
            for event in span.events:
                if event.name == "metadata" and event.attributes.get("finish_type") == "tool_call":
                    return span
    return None

@pytest.mark.skipif(not MISTRAL_API_KEY, reason="MISTRAL_API_KEY not set")
def test_finish_reason_tool_calls_with_entity_validation(setup):
    """Test finish_reason == 'tool_calls' and validate entity.3 attributes."""
    client = Mistral(api_key=MISTRAL_API_KEY)
    
    # Tool definition for weather function
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    
    resp = client.chat.complete(
        model=MODEL,
        max_tokens=1000,
        tools=tools,
        messages=[{"role": "user", "content": "What is the weather in Paris, France?"}],
    )

    assert resp.choices[0].finish_reason in ("tool_calls", "stop")
    logger.info("tool_calls finish_reason: %s", resp.choices[0].finish_reason)

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"

    found_entity3 = False
    inference_span = find_inference_span_with_tool_call(spans)
    if inference_span:
        # Validate entity.3 attributes (tool information)
        entity_3_name = inference_span.attributes.get("entity.3.name")
        entity_3_type = inference_span.attributes.get("entity.3.type")

        logger.info(f"entity.3.name: {entity_3_name}")
        logger.info(f"entity.3.type: {entity_3_type}")

        # Validate entity.3 attributes are present and correct
        assert entity_3_name == "get_current_weather", f"Expected entity.3.name='get_current_weather', got '{entity_3_name}'"
        assert entity_3_type == "tool.function", f"Expected entity.3.type='tool.function', got '{entity_3_type}'"
        found_entity3 = True

    assert found_entity3, "Entity 3 not found"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])