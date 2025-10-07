import asyncio
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
from mistralai import Mistral, models
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.utils import logger
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="generic_mistral_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            # wrapper_methods=MISTRAL_METHODS,
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_mistral_api_sample(setup):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    response = client.chat.complete(
        model="mistral-small",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )

    time.sleep(5)

    spans = setup.get_captured_spans()
    logger.info(f"Captured {len(spans)} spans")
    
    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected at least one inference span"

    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.mistral",
            model_name="mistral-small",
            model_type="model.llm.mistral-small",
            check_metadata=True,
            check_input_output=True,
        )
    assert len(inference_spans) == 1, "Expected exactly one inference span"


    validate_inference_span_events(
        span=inference_spans[0],
        expected_event_count=3,
        input_patterns=[
            r"^\{\"system\": \".+\"\}$",
            r"^\{\"user\": \".+\"\}$",
        ],
        output_pattern=r"^\{\"assistant\": \".+\"\}$",
        metadata_requirements={
            "completion_tokens": int,
            "prompt_tokens": int,
            "total_tokens": int,
        },
    )
    
    workflow_span = find_span_by_type(spans, "workflow")
    assert workflow_span is not None, "Expected workflow span"

    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "generic_mistral_1"
    assert workflow_span.attributes["workflow.name"] == "generic_mistral_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.mistral"

@pytest.mark.asyncio
async def test_mistral_api_sample_async(setup):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    response = await client.chat.complete_async(   # <-- fixed method name
        model="mistral-small",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )

    # give some time for spans to flush
    await asyncio.sleep(5)

    spans = setup.get_captured_spans()
    logger.info(f"Captured {len(spans)} spans")

    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected at least one inference span"

    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.mistral",
            model_name="mistral-small",
            model_type="model.llm.mistral-small",
            check_metadata=True,
            check_input_output=True,
        )

        # Add assertion for span.subtype
        assert "span.subtype" in span.attributes, "Expected span.subtype attribute to be present"
        assert span.attributes.get("span.subtype") in ["turn_end", "tool_call", "delegation"]
    assert len(inference_spans) == 1, "Expected exactly one inference span"

    validate_inference_span_events(
        span=inference_spans[0],
        expected_event_count=3,
        input_patterns=[
            r"^\{\"system\": \".+\"\}$",
            r"^\{\"user\": \".+\"\}$",
        ],
        output_pattern=r"^\{\"assistant\": \".+\"\}$",
        metadata_requirements={
            "completion_tokens": int,
            "prompt_tokens": int,
            "total_tokens": int,
        },
    )

    workflow_span = find_span_by_type(spans, "workflow")
    assert workflow_span is not None, "Expected workflow span"
    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "generic_mistral_1"
    assert workflow_span.attributes["workflow.name"] == "generic_mistral_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.mistral"


def test_mistral_invalid_api_key(setup):
    try:
        client = Mistral(api_key="invalid_key_123")
        client.chat.complete(
            model="mistral-small",
            messages=[{"role": "user", "content": "test"}],
        )
    except models.SDKError as e:
        # e.status_code is correct
        assert e.status_code == 401
        # The response body contains the actual "Unauthorized" text
        assert '"Unauthorized"' in e.body

    # Wait for spans to be flushed
    time.sleep(5)
    spans = setup.get_captured_spans()

    for span in spans:
        if span.attributes.get("span.type") in ["inference", "inference.framework"]:
            events = [e for e in span.events if e.name == "data.output"]
            assert len(events) > 0
            # Span should have ERROR status
            assert span.status.status_code.value == 2

            # Current error_code in span is just "error"
            error_code = events[0].attributes.get("error_code")
            assert error_code == "error"

            response = events[0].attributes.get("response")
            assert response is None or response == ""


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])