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
from huggingface_hub import AsyncInferenceClient, InferenceClient
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.metamodel.hugging_face.methods import (
    HUGGING_FACE_METHODS,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from config.conftest import temporary_env_var

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    instrumentor = None
    try:
        span_processors = [
            BatchSpanProcessor(file_exporter),
            SimpleSpanProcessor(custom_exporter),
        ]
        instrumentor = setup_monocle_telemetry(
            workflow_name="generic_hf_1",
            span_processors=span_processors,
            wrapper_methods=[]
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_huggingface_api_sample(setup):
    baseline_count = len(setup.get_captured_spans())
    client = InferenceClient(api_key=os.getenv("HUGGING_FACE_API_KEY"))
    response = client.chat_completion(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )

    time.sleep(5)  # wait for spans

    spans = setup.get_captured_spans()[baseline_count:]
    logger.info(f"Captured {len(spans)} spans")
    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference") or find_spans_by_type(spans, "inference.framework")
    assert inference_spans, "Expected at least one inference span"

    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.huggingface",
            model_name="openai/gpt-oss-120b",
            model_type="model.llm.openai/gpt-oss-120b",
            check_metadata=True,
            check_input_output=True,
        )

        # Add assertion for span.subtype
        assert "span.subtype" in span.attributes, "Expected span.subtype attribute to be present"
        assert span.attributes.get("span.subtype") in ["turn_end", "tool_call", "delegation"]

    logger.info("\n--- Captured span types ---")
    for span in spans:
        logger.info(f"{span.attributes.get('span.type', 'NO_SPAN_TYPE')} - {span.name}")


    inference_spans = [s for s in inference_spans if s.attributes.get("span.type", "") == "inference"]
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
    assert workflow_span is not None
    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "generic_hf_1"
    assert workflow_span.attributes["workflow.name"] == "generic_hf_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.huggingface"

@pytest.mark.asyncio
async def test_huggingface_api_async_sample(setup):
    baseline_count = len(setup.get_captured_spans())
    # Use AsyncInferenceClient
    client = AsyncInferenceClient(api_key=os.getenv("HUGGING_FACE_API_KEY"))

    # Use await to call the async chat_completion method
    response = await client.chat_completion(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )

    time.sleep(5)  # wait for spans

    spans = setup.get_captured_spans()[baseline_count:]
    logger.info(f"Captured {len(spans)} spans")
    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference") or find_spans_by_type(spans, "inference.framework")
    assert inference_spans, "Expected at least one inference span"

    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.huggingface",
            model_name="openai/gpt-oss-120b",
            model_type="model.llm.openai/gpt-oss-120b",
            check_metadata=True,
            check_input_output=True,
        )

        # Add assertion for span.subtype
        assert "span.subtype" in span.attributes, "Expected span.subtype attribute to be present"
        assert span.attributes.get("span.subtype") in ["turn_end", "tool_call", "delegation"]

    logger.info("\n--- Captured span types ---")
    for span in spans:
        logger.info(f"{span.attributes.get('span.type', 'NO_SPAN_TYPE')} - {span.name}")

    inference_spans = [s for s in inference_spans if s.attributes.get("span.type", "") == "inference"]
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
    assert workflow_span is not None
    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "generic_hf_1"
    assert workflow_span.attributes["workflow.name"] == "generic_hf_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.huggingface"


@pytest.mark.asyncio
async def test_huggingface_api_async_streaming_sample(setup):
    baseline_count = len(setup.get_captured_spans())
    client = AsyncInferenceClient(api_key=os.getenv("HUGGING_FACE_API_KEY"))

    stream = await client.chat_completion(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
        stream=True,
    )

    # Consume streaming chunks so telemetry can capture full call lifecycle.
    async for _chunk in stream:
        pass

    time.sleep(5)  # wait for spans

    spans = setup.get_captured_spans()[baseline_count:]
    logger.info(f"Captured {len(spans)} spans")
    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference") or find_spans_by_type(spans, "inference.framework")
    assert inference_spans, "Expected at least one inference span"

    inference_spans = [s for s in inference_spans if s.attributes.get("span.type", "") == "inference"]
    assert len(inference_spans) == 1, "Expected exactly one inference span"

    span = inference_spans[0]
    verify_inference_span(
        span=span,
        entity_type="inference.huggingface",
        model_name="openai/gpt-oss-120b",
        model_type="model.llm.openai/gpt-oss-120b",
        check_metadata=True,
        check_input_output=False,
    )

    assert "span.subtype" in span.attributes, "Expected span.subtype attribute to be present"
    assert span.attributes.get("span.subtype") in ["turn_end", "tool_call", "delegation"]

    workflow_span = find_span_by_type(spans, "workflow")
    assert workflow_span is not None
    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "generic_hf_1"
    assert workflow_span.attributes["workflow.name"] == "generic_hf_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.huggingface"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
