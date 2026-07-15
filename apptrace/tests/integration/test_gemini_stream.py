import logging
import os
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)
from google import genai
from google.genai import types
from monocle_apptrace.instrumentation.common.instrumentor import (
    setup_monocle_telemetry,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from monocle_apptrace.exporters.file_exporter import FileSpanExporter


logger = logging.getLogger(__name__)



@pytest.fixture(scope="module")
def setup():
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="gemini_stream_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter), BatchSpanProcessor(FileSpanExporter())],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_gemini_model_sample(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a cat. Your name is Neko."
        ),
        contents="How many nails do you have?",
    )
    # Consume the stream so the span is finalised and spans are exported.
    for chunk in response:
        pass
    time.sleep(5)
    # logger.info(response.text)
    spans = setup.get_captured_spans()
    check_span(spans)
    setup.reset()


def test_gemini_chat_sample(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    chat = client.chats.create(model="gemini-2.5-flash")

    # First message (streaming)
    response_text = ""
    for chunk in chat.send_message_stream("I have 2 dogs in my house."):
        if chunk.text:
            response_text += chunk.text
            logger.info(chunk.text)

    # Second message (streaming)
    response_text = ""
    for chunk in chat.send_message_stream("How many paws are in my house?"):
        if chunk.text:
            response_text += chunk.text
            logger.info(chunk.text)

    time.sleep(5)

    logger.info(response_text)

    spans = setup.get_captured_spans()
    check_span_chat(spans)
    setup.reset()

def check_span(spans):
    """Verify spans using flexible utilities."""
    # Find workflow span
    workflow_span = find_span_by_type(spans, "workflow")
    assert workflow_span is not None, "Expected to find workflow span"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        # Also check for inference.framework spans
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify each inference span
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.gemini",
            model_name="gemini-2.5-flash",
            model_type="model.llm.gemini-2.5-flash",
            check_metadata=True,
            check_input_output=True,
        )
    assert (
        len(inference_spans) == 1
    ), "Expected exactly one inference span for the LLM call"

    # Validate events using the generic function with regex patterns
    validate_inference_span_events(
        span=inference_spans[0],
        expected_event_count=3,
        input_patterns=[
            r"^\{\"system\": \".+\"\}$",  # Pattern for system message
            r"^\{\"user\": \".+\"\}$",  # Pattern for user message
        ],
        output_pattern=r"^\{\"model\": \".+\"\}$",  # Pattern for AI response
        metadata_requirements={
            "completion_tokens": int,
            "prompt_tokens": int,
            "total_tokens": int,
        },
    )


def check_span_chat(spans):
    """Verify spans using flexible utilities."""
    # Find workflow span
    workflow_span = find_span_by_type(spans, "workflow")
    assert workflow_span is not None, "Expected to find workflow span"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        # Also check for inference.framework spans
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify each inference span
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.gemini",
            model_name="gemini-2.5-flash",
            model_type="model.llm.gemini-2.5-flash",
            check_metadata=True,
            check_input_output=True,
        )
    assert (
        len(inference_spans) == 2
    ), "Expected exactly two inference spans for the LLM call"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
