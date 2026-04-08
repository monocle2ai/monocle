import logging
import time
import os

import anthropic
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    verify_inference_span,
)
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.utils import logger
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

logger = logging.getLogger(__name__)
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL")


@pytest.fixture(scope="module")
def setup():
    try:
        custom_exporter = CustomConsoleSpanExporter()
        file_exporter = FileSpanExporter()
        span_processors = [
            BatchSpanProcessor(file_exporter),
            SimpleSpanProcessor(custom_exporter)
        ]
        instrumentor = setup_monocle_telemetry(
            workflow_name="anthropic_streaming_app",
            span_processors=span_processors,
            wrapper_methods=[
            ])
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_anthropic_streaming_sample(setup):
    """Test Anthropic streaming API with monocle instrumentation."""
    client = anthropic.Anthropic()

    # Collect streamed text
    full_response = ""

    # Use create(stream=True): this path is wrapped by monocle anthropic instrumentation.
    stream = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        temperature=0.7,
        system="You are a helpful assistant to answer questions about coffee.",
        messages=[
            {"role": "user", "content": "What is a cappuccino?"}
        ],
        stream=True,
    )
    for event in stream:
        if getattr(event, "type", None) == "content_block_delta" and hasattr(event, "delta"):
            text = getattr(event.delta, "text", "")
            if text:
                full_response += text
                logger.debug("Streamed text: %s", text)

    # Log the complete response
    logger.info("Complete streamed response:\n")
    logger.info(full_response)

    # Verify we got a response
    assert full_response, "No response received from streaming API"
    assert len(full_response) > 0, "Response should not be empty"

    time.sleep(5)
    spans = setup.get_captured_spans()

    assert len(spans) > 0, "No spans captured for the Anthropic streaming sample"

    # Find inference spans
    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        # Also check for inference.framework spans
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify each inference span
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.anthropic",
            model_name=ANTHROPIC_MODEL,
            model_type="model.llm." + ANTHROPIC_MODEL,
            check_metadata=True,
            check_input_output=True,
        )
        # Add assertion for span.subtype
        assert "span.subtype" in span.attributes, "Expected span.subtype attribute to be present"
        assert span.attributes.get("span.subtype") in ["turn_end", "tool_call", "delegation"]

    assert (
        len(inference_spans) == 1
    ), "Expected exactly one inference span for the LLM call"

    # Verify workflow span
    workflow_span = find_span_by_type(spans, "workflow")

    assert workflow_span is not None, "Expected to find workflow span"

    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "anthropic_streaming_app"
    assert workflow_span.attributes["entity.1.type"] == "workflow.anthropic"


def test_anthropic_streaming_with_metadata(setup):
    """Test that streaming responses include proper metadata."""
    client = anthropic.Anthropic()

    full_response = ""

    # Stream with metadata collection through create(stream=True)
    stream = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=256,
        system="You are a helpful assistant.",
        messages=[
            {"role": "user", "content": "What is espresso?"}
        ],
        stream=True,
    )
    for event in stream:
        if getattr(event, "type", None) == "content_block_delta" and hasattr(event, "delta"):
            text = getattr(event.delta, "text", "")
            if text:
                full_response += text

    logger.info("Streamed response collected")

    time.sleep(5)
    spans = setup.get_captured_spans()

    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected inference span"

    # Verify metadata was captured
    for span in inference_spans:
        # Check for metadata event
        metadata_events = [e for e in span.events if e.name == "metadata"]
        assert len(metadata_events) > 0, "Expected metadata event in span"

        # Verify metadata attributes exist
        for event in metadata_events:
            assert "completion_tokens" in event.attributes or len(event.attributes) >= 0
            assert "prompt_tokens" in event.attributes or len(event.attributes) >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
