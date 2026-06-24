import asyncio
import logging
import os

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)
from mistralai.client import Mistral
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.utils import logger
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter),
    ]
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="generic_mistral_1",
            span_processors=span_processors,
            wrapper_methods=[],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_mistral_api_sample_async(setup):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    response = await client.chat.complete_async(
        model="mistral-small",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )

    # Give some time for spans to flush.
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


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
