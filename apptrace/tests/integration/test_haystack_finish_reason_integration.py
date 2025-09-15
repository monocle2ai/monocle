"""
Integration test for Haystack finish_reason using real Haystack integrations.
Tests various Haystack providers and scenarios: OpenAI, tool calls, etc.

Requirements:
- Set OPENAI_API_KEY in your environment
- Requires haystack, haystack-integrations

Run with: pytest tests/integration/test_haystack_finish_reason_integration.py
"""
import os
from tokenize import generate_tokens

import pytest
from haystack.utils import Secret
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from haystack.dataclasses import ChatMessage
from monocle_apptrace.instrumentation.metamodel.finish_types import map_haystack_finish_reason_to_finish_type, FinishType
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from tests.common.custom_exporter import CustomConsoleSpanExporter
pytestmark = pytest.mark.integration

# Setup telemetry
custom_exporter = CustomConsoleSpanExporter()
setup_monocle_telemetry(
    workflow_name="haystack_integration_tests",
    span_processors=[SimpleSpanProcessor(custom_exporter)],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

def find_inference_span_and_event_attributes(spans, event_name="metadata"):
    """Find inference span and return event attributes."""
    for span in reversed(spans):  # Usually the last span is the inference span
        span_type = span.attributes.get("span.type")
        if span_type in ("inference.framework", "inference"):
            for event in span.events:
                if event.name == event_name:
                    return event.attributes
    return None

@pytest.fixture(autouse=True)
def clear_exporter_before_test():
    """Clear exporter before each test."""
    custom_exporter.reset()


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or haystack not available"
)
def test_haystack_openai_finish_reason_stop():
    from haystack.components.generators import OpenAIGenerator
    generation_kwargs = {'max_tokens': 50, 'temperature': 0.0}
    generator = OpenAIGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model="gpt-3.5-turbo",generation_kwargs=generation_kwargs)
    result = generator.run("Say hello in one word.")
    print(f"OpenAI Haystack response: {result}")
    # time.sleep(5)  # Allow time for spans to be captured
    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"

    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "data.output event not found in inference span"

    # Check that finish_reason and finish_type are captured
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")

    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")

    assert finish_reason in ["stop", None]
    if finish_reason:
        assert finish_type == FinishType.SUCCESS.value


@pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set or haystack-anthropic not available"
)
def test_haystack_anthropic_finish_reason():
    """Test finish_reason with Haystack Anthropic integration."""
    from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
    generator  = AnthropicChatGenerator(model="claude-3-5-sonnet-20240620",
                                       generation_kwargs={
                                           "max_tokens": 50,
                                           "temperature": 0.0,
                                       }
    )
    messages = [ChatMessage.from_system("You are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("Say hello briefly.")]
    response = generator.run(messages=messages)
    print(f"Anthropic Haystack response: {response}")

    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"

    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "data.output event not found in inference span"

    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")

    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")

    # Anthropic typically uses "end_turn" for normal completion
    if finish_reason:
        assert finish_reason in ["end_turn", "stop_sequence", "stop"]
        assert finish_type == "success"

@pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set or haystack-anthropic not available"
)
def test_haystack_anthropic_finish_reason_max_tokens():
    """Test finish_reason with Haystack Anthropic integration."""
    from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
    generator  = AnthropicChatGenerator(model="claude-3-5-sonnet-20240620",
                                       generation_kwargs={
                                           "max_tokens": 1,
                                           "temperature": 0.0,
                                       }
    )
    messages = [ChatMessage.from_system("You are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("Write a detailed explanation of quantum computing.")]
    response = generator.run(messages=messages)
    print(f"Anthropic Haystack Truncate response: {response}")

    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"

    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "data.output event not found in inference span"

    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")

    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")

    # Should be max_tokens/truncated when hitting token limit
    if finish_reason:
        assert finish_reason in ["max_tokens", "length"]
        assert finish_type == "truncated"

@pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set or haystack-anthropic not available"
)
def test_haystack_anthropic_generator_finish_reason_max_tokens():
    """Test finish_reason with Haystack Anthropic integration."""
    from haystack_integrations.components.generators.anthropic import AnthropicGenerator
    generator  = AnthropicGenerator(model="claude-3-5-sonnet-20240620",
                                       generation_kwargs={
                                           "max_tokens": 1,
                                           "temperature": 0.0,
                                       }
    )
    messages = [ChatMessage.from_system("You are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("Write a detailed explanation of quantum computing.")]
    response = generator.run("Write a detailed explanation of quantum computing.")
    print(f"Anthropic Haystack Truncate response: {response}")

    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"

    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "data.output event not found in inference span"

    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")

    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")

    # Should be max_tokens/truncated when hitting token limit
    if finish_reason:
        assert finish_reason in ["max_tokens", "length"]
        assert finish_type == "truncated"

@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or haystack not available"
)
def test_haystack_openai_finish_reason_length():
    from haystack.components.generators import OpenAIGenerator
    generation_kwargs = {'max_tokens': 1, 'temperature': 0.0}
    generator = OpenAIGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model="gpt-3.5-turbo",generation_kwargs=generation_kwargs)
    result = generator.run("Write a long story about a dragon and a princess.")
    print(f"OpenAI Haystack truncated response: {result}")

    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"

    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "data.output event not found in inference span"

    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")

    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")

    if finish_reason:
        assert finish_reason in ["length", "max_tokens"]
        assert finish_type == FinishType.TRUNCATED.value

def test_haystack_finish_reason_extraction_fallback():
    # Test fallback logic for missing finish_reason
    # This test doesn't require API keys as it tests the fallback logic
    from src.monocle_apptrace.instrumentation.metamodel.haystack._helper import extract_finish_reason

    # Mock a Haystack response without explicit finish_reason
    from types import SimpleNamespace

    mock_response = SimpleNamespace()  # Empty response
    arguments = {
        "exception": None,
        "result": mock_response
    }

    result = extract_finish_reason(arguments)
    assert result == "stop"  # Should default to success case

    # Test with exception
    arguments_with_exception = {
        "exception": Exception("Test error"),
        "result": None
    }

    result = extract_finish_reason(arguments_with_exception)
    assert result == "error"

def test_haystack_finish_reason_mapping_edge_cases():
    # Case insensitive mapping
    assert map_haystack_finish_reason_to_finish_type("STOP") == FinishType.SUCCESS.value
    assert map_haystack_finish_reason_to_finish_type("Stop") == FinishType.SUCCESS.value
    assert map_haystack_finish_reason_to_finish_type("MAX_TOKENS") == FinishType.TRUNCATED.value
    # Pattern matching
    assert map_haystack_finish_reason_to_finish_type("completion_stopped") == FinishType.SUCCESS.value
    assert map_haystack_finish_reason_to_finish_type("token_limit_reached") == FinishType.TRUNCATED.value
    assert map_haystack_finish_reason_to_finish_type("safety_filter_applied") == FinishType.CONTENT_FILTER.value
    assert map_haystack_finish_reason_to_finish_type("unexpected_error") == FinishType.ERROR.value
    # Unknown reasons
    assert map_haystack_finish_reason_to_finish_type("unknown_reason") is None
    assert map_haystack_finish_reason_to_finish_type(None) is None
    assert map_haystack_finish_reason_to_finish_type("") is None

@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or haystack not available"
)

def test_haystack_openai_finish_reason_content_filter():
    from haystack.components.generators import OpenAIGenerator
    generator = OpenAIGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model="gpt-3.5-turbo")
    result = generator.run("Describe how to make a dangerous substance.")
    print(f"OpenAI Haystack content filter response: {result}")
    spans = custom_exporter.get_captured_spans()
    output_event_attrs = find_inference_span_and_event_attributes(spans)

    assert output_event_attrs, "data.output event not found in inference span"

    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")

    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")

    if finish_reason:
        assert finish_reason in ["content_filter", "stop"]
        if finish_reason == "content_filter":
            assert finish_type == FinishType.CONTENT_FILTER.value
        elif finish_reason == "stop":
            assert finish_type == FinishType.SUCCESS.value


def test_haystack_anthropic_finish_reason_content_filter():
    from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
    generator = AnthropicChatGenerator(model="claude-3-5-sonnet-20240620",
                                    )
    messages = [ChatMessage.from_system("You are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("Describe how to make a dangerous substance.")]
    response = generator.run(messages=messages)
    print(f"Anthropic Haystack content filter response: {response}")

    spans = custom_exporter.get_captured_spans()
    output_event_attrs = find_inference_span_and_event_attributes(spans)

    assert output_event_attrs, "data.output event not found in inference span"

    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")

    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")

    # Anthropic may use "refusal" or other safety-related finish reasons
    if finish_reason:
        if finish_reason in ["refusal", "safety", "content_filter"]:
            assert finish_type in ["content_filter", "refusal"]
        elif finish_reason in ["end_turn", "stop"]:
            assert finish_type == "success"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])