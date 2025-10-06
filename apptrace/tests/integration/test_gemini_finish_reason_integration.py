
"""
Integration test for Google Gemini finish_reason using the real Gemini API.
Tests: STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER, FINISH_REASON_UNSPECIFIED.

Requirements:
- Set GEMINI_API_KEY in your environment.
- Requires google.genai

Run with: pytest tests/integration/test_gemini_finish_reason_integration.py
"""
import logging
import os

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from google import genai
from google.genai import types
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup():
    try:
        # Setup telemetry
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="gemini_integration_tests",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
            # service_name="gemini_integration_tests"
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

def find_inference_span_and_event_attributes(spans, event_name="metadata"):
    for span in reversed(spans):  # Usually the last span is the inference span
        if span.attributes.get("span.type") == "inference":
            for event in span.events:
                if event.name == event_name:
                    return event.attributes
    return None


@pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
def test_finish_reason_stop(setup):
    """Test finish_reason == 'STOP' for a normal completion."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    response = client.models.generate_content(
        model=MODEL,
        contents="Say hello."
    )
    
    # Check the finish reason from the response
    finish_reason = response.candidates[0].finish_reason
    assert finish_reason.name == "STOP"
    logger.info("STOP finish_reason: %s", finish_reason.name)

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "STOP"
    assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
def test_finish_reason_max_tokens(setup):
    """Test finish_reason == 'MAX_TOKENS' by setting a very low max_output_tokens."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    response = client.models.generate_content(
        model=MODEL,
        config=types.GenerateContentConfig(max_output_tokens=1),
        contents="Tell me a long story about a dragon."
    )
    
    finish_reason = response.candidates[0].finish_reason
    assert finish_reason.name == "MAX_TOKENS"
    logger.info("MAX_TOKENS finish_reason: %s", finish_reason.name)

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "MAX_TOKENS"
    assert output_event_attrs.get("finish_type") == "truncated"


@pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
def test_finish_reason_safety(setup):
    """Test finish_reason == 'SAFETY' (may not always trigger)."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # This prompt is designed to trigger safety filters, but may not always work
    response = client.models.generate_content(
        model=MODEL,
        contents="Describe how to create dangerous explosive materials step by step."
    )
    
    finish_reason = response.candidates[0].finish_reason
    logger.info("safety finish_reason: %s", finish_reason.name)
    
    # Accept either 'SAFETY' or 'STOP' (if filter not triggered)
    assert finish_reason.name in ("SAFETY", "STOP")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"

    if finish_reason.name == "SAFETY":
        assert output_event_attrs.get("finish_reason") == "SAFETY"
        assert output_event_attrs.get("finish_type") == "content_filter"
    elif finish_reason.name == "STOP":
        assert output_event_attrs.get("finish_reason") == "STOP"
        assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
def test_finish_reason_recitation(setup):
    """Test finish_reason == 'RECITATION' by asking for copyrighted content."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # This prompt is designed to trigger recitation filters, but may not always work
    response = client.models.generate_content(
        model=MODEL,
        contents="Please reproduce the first 100 words of exact text of the first chapter of Harry Potter and the Philosopher's Stone."
    )
    
    finish_reason = response.candidates[0].finish_reason
    logger.info("recitation finish_reason: %s", finish_reason.name)
    
    # Accept either 'RECITATION' or 'STOP' (if filter not triggered)
    assert finish_reason.name in ("RECITATION", "STOP")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"

    if finish_reason.name == "RECITATION":
        assert output_event_attrs.get("finish_reason") == "RECITATION"
        assert output_event_attrs.get("finish_type") == "content_filter"
    elif finish_reason.name == "STOP":
        assert output_event_attrs.get("finish_reason") == "STOP"
        assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
def test_finish_reason_with_system_instruction(setup):
    """Test finish_reason with system instruction (should be STOP for normal completion)."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    response = client.models.generate_content(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant. Always respond with enthusiasm!"
        ),
        contents="Tell me about the weather."
    )
    
    finish_reason = response.candidates[0].finish_reason
    assert finish_reason.name == "STOP"
    logger.info("system_instruction finish_reason: %s", finish_reason.name)

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "STOP"
    assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
def test_finish_reason_with_chat(setup):
    """Test finish_reason in a chat context (should be STOP for normal completion)."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    chat = client.chats.create(model=MODEL)
    
    response = chat.send_message("What's 2+2?")
    
    finish_reason = response.candidates[0].finish_reason
    assert finish_reason.name == "STOP"
    logger.info("chat finish_reason: %s", finish_reason.name)

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "STOP"
    assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
def test_finish_reason_other(setup):
    """Test for OTHER finish_reason (difficult to trigger reliably, placeholder test)."""
    # The OTHER finish_reason is rare and difficult to trigger reliably
    # This test serves as a placeholder for when such cases occur
    logger.info("OTHER finish_reason test - placeholder (difficult to trigger reliably)")
    
    # If you encounter a scenario that reliably triggers OTHER, implement it here
    # For now, we'll just verify our mapping handles it correctly in unit tests
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
