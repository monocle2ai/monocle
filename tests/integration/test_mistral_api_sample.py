import asyncio
import os
import time
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from mistralai import Mistral, models
from monocle_apptrace.instrumentation.metamodel.mistral.methods import MISTRAL_METHODS

from tests.common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(autouse=True)
def clear_spans():
    """Clear spans before each test"""
    custom_exporter.reset()
    yield

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="generic_mistral_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        # wrapper_methods=MISTRAL_METHODS,
    )

@pytest.mark.integration()
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

    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")

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
