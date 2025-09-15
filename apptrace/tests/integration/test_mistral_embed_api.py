import os
import time
import pytest
from mistralai import Mistral, models
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.utils import logger
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter

from tests.common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    verify_inference_span,
    verify_embedding_span,   # <-- add this
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
        workflow_name="generic_mistral_embed",
        span_processors=[BatchSpanProcessor(custom_exporter)],
    )

# -----------------------------
# SYNC TEST
# -----------------------------
@pytest.mark.integration()
def test_mistral_embeddings_sample(setup):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=["Hello world!", "Coffee is amazing."]
    )

    time.sleep(5)
    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")
    # Check embedding response
    assert len(response.data) == 2, "Expected embeddings for 2 inputs"
    assert len(response.data[0].embedding) == 1024, "Embedding dimension mismatch"

    # Check spans
    relevant_spans = (
        find_spans_by_type(spans, "inference")
        + find_spans_by_type(spans, "inference.framework")
        + find_spans_by_type(spans, "embedding")
    )
    
    assert len(relevant_spans) > 0, "Expected spans for embeddings"

    for span in relevant_spans:
        if span.attributes["span.type"] == "embedding":
            verify_embedding_span(span=span, model_name="mistral-embed")

        else:
            verify_inference_span(
                span=span, 
                entity_type="inference.mistral",
                model_name="mistral-embed",
                model_type="model.embedding.mistral",
                check_metadata=True,
                check_input_output=True,
            )

    workflow_span = find_span_by_type(spans, "workflow")
    assert workflow_span is not None, "Expected workflow span"
    assert workflow_span.attributes["entity.1.name"] == "generic_mistral_embed"
    assert workflow_span.attributes["entity.1.type"] == "workflow.mistral"

# -----------------------------
# INVALID API KEY TEST
# -----------------------------
@pytest.mark.integration()
def test_mistral_embeddings_invalid_api_key(setup):
    try:
        client = Mistral(api_key="invalid_key_123")
        client.embeddings.create(
            model="mistral-embed",
            inputs=["test"]
        )
    except models.SDKError as e:
        assert e.status_code == 401
        assert '"Unauthorized"' in e.body

    time.sleep(5)
    spans = custom_exporter.get_captured_spans()

    for span in spans:
        if span.attributes.get("span.type") in ["inference", "inference.framework"]:
            events = [e for e in span.events if e.name == "data.output"]
            assert len(events) > 0
            assert span.status.status_code.value == 2  # ERROR
            error_code = events[0].attributes.get("error_code")
            assert error_code == "error"
