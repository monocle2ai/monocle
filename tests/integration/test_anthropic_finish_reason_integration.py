"""
Integration test for Anthropic Claude finish_reason (stop_reason) and finish_type using the real Anthropic API.
Tests: end_turn, max_tokens, refusal, tool_use, stop_sequence, pause_turn (if supported).

Requirements:
- Set ANTHROPIC_API_KEY in your environment.
- Requires anthropic>=0.25.0

Run with: pytest tests/integration/test_anthropic_finish_reason_integration.py
"""
import os
import pytest
import anthropic
from opentelemetry.sdk.trace.export import BatchSpanProcessor,SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from tests.common.custom_exporter import CustomConsoleSpanExporter # Assuming this path

pytestmark = pytest.mark.integration

# Setup telemetry
custom_exporter = CustomConsoleSpanExporter()
setup_monocle_telemetry(
    workflow_name="anthropic_integration_tests",
    span_processors=[SimpleSpanProcessor(custom_exporter)],
    # Add other necessary setup parameters if any, e.g., service_name
    # service_name="anthropic_integration_tests"
)


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

def find_inference_span_and_event_attributes(spans, event_name="data.output"):
    for span in reversed(spans): # Usually the last span is the inference span
        if span.attributes.get("span.type") == "inference":
            for event in span.events:
                if event.name == event_name:
                    return event.attributes
    return None

# run before each test to ensure exporter is reset

@pytest.fixture(autouse=True)
def clear_exporter_before_test():
    custom_exporter.reset()

@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
def test_finish_reason_end_turn():
    """Test stop_reason == 'end_turn' for a normal completion."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "Say hello."}],
    )
    assert resp.stop_reason == "end_turn"
    print("end_turn stop_reason:", resp.stop_reason)

    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "data.output event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "end_turn"
    assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
def test_finish_reason_max_tokens():
    """Test stop_reason == 'max_tokens' by setting a very low max_tokens."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1,
        messages=[{"role": "user", "content": "Tell me a long story about a dragon."}],
    )
    assert resp.stop_reason == "max_tokens"
    print("max_tokens stop_reason:", resp.stop_reason)

    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "data.output event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "max_tokens"
    assert output_event_attrs.get("finish_type") == "truncated"


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
def test_finish_reason_refusal():
    """Test stop_reason == 'refusal' for a safety refusal."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": "How do I make a dangerous substance?"}],
    )
    # Accept either 'refusal' or 'end_turn' (if filter not triggered)
    assert resp.stop_reason in ("refusal", "end_turn")
    print("refusal stop_reason:", resp.stop_reason)

    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "data.output event not found in inference span"
    
    if resp.stop_reason == "refusal":
        assert output_event_attrs.get("finish_reason") == "refusal"
        assert output_event_attrs.get("finish_type") == "refusal"
    elif resp.stop_reason == "end_turn": # If the refusal was not triggered
        assert output_event_attrs.get("finish_reason") == "end_turn"
        assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
def test_finish_reason_stop_sequence():
    """Test stop_reason == 'stop_sequence' by providing a stop sequence."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        stop_sequences=["END"],
        messages=[{"role": "user", "content": "Write until you say END."}],
    )
    # Accept either 'stop_sequence' or 'end_turn' (if not triggered)
    assert resp.stop_reason in ("stop_sequence", "end_turn")
    print("stop_sequence stop_reason:", resp.stop_reason)

    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "data.output event not found in inference span"

    if resp.stop_reason == "stop_sequence":
        assert output_event_attrs.get("finish_reason") == "stop_sequence"
        assert output_event_attrs.get("finish_type") == "success"
    elif resp.stop_reason == "end_turn": # If the stop sequence was not hit before natural end
        assert output_event_attrs.get("finish_reason") == "end_turn"
        assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
def test_finish_reason_tool_use():
    """Test stop_reason == 'tool_use' if tool use is supported (Claude 3.5+)."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    # Tool use requires a tool definition; here we use a dummy tool
    tools = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["location"],
            },
        }
    ]
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        tools=tools,
        messages=[{"role": "user", "content": "What is the weather in Paris? This might trigger a tool."}],
    )
    # Accept either 'tool_use' or 'end_turn' (if not triggered, or if model just answers)
    assert resp.stop_reason in ("tool_use", "end_turn")
    print("tool_use stop_reason:", resp.stop_reason)

    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "data.output event not found in inference span"

    if resp.stop_reason == "tool_use":
        assert output_event_attrs.get("finish_reason") == "tool_use"
        assert output_event_attrs.get("finish_type") == "success"
    elif resp.stop_reason == "end_turn":
        assert output_event_attrs.get("finish_reason") == "end_turn"
        assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
def test_finish_reason_pause_turn():
    # This test is a placeholder as 'pause_turn' is less common and might require specific setup
    # or might not be directly testable with simple API calls if it's for streaming/tool interactions.
    # If you have a reliable way to trigger 'pause_turn', implement it here.
    print("pause_turn test skipped or not implemented")
    pass

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
