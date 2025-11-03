import logging
import os
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    verify_embedding_span,  # <-- add this
    verify_inference_span,
)
from mistralai import Mistral, models
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.utils import logger
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from config.conftest import temporary_env_var

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="generic_mistral_embed",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

# -----------------------------
# SYNC TEST
# -----------------------------
def test_mistral_embeddings_sample(setup):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=["Hello world!", "Coffee is amazing."]
    )

    time.sleep(5)
    spans = setup.get_captured_spans()
    logger.info(f"Captured {len(spans)} spans")
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
def test_mistral_embeddings_invalid_api_key(setup):
    """Test Mistral embeddings API with invalid API key using temporary environment variable"""
    with temporary_env_var("MISTRAL_API_KEY", "invalid_key_123"):
        client = Mistral()  # picks token from env
        try:
            client.embeddings.create(
                model="mistral-embed",
                inputs=["test"]
            )
        except Exception as e:
            # Accept 401, 403, Unauthorized, Forbidden errors
            error_str = str(e)
            assert (
                "401" in error_str
                or "403" in error_str
                or "Unauthorized" in error_str
                or "Forbidden" in error_str
            ), f"Unexpected error: {error_str}"

        time.sleep(5)
        spans = setup.get_captured_spans()
        for span in spans:
            logger.info(f"SPAN: {span.name}")
            for e in span.events:
                logger.info(f" EVENT: {e.name} {e.attributes}")

        for span in spans:
            if span.attributes.get("span.type") in ["inference", "inference.framework"]:
                assert "span.subtype" in span.attributes
                assert span.attributes.get("span.subtype") in ["turn_end", "tool_call", "delegation"]
                events = [e for e in span.events if e.name == "data.output"]
                assert len(events) > 0
                assert span.status.status_code.value == 2
                error_code = events[0].attributes.get("error_code")
                assert error_code == "error"
                response = events[0].attributes.get("response")
                assert response is None or response == ""
