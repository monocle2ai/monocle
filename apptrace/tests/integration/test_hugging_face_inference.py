import asyncio
import os
import time
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from tests.common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.metamodel.hugging_face.methods import HUGGING_FACE_METHODS
from huggingface_hub import InferenceClient
from huggingface_hub import AsyncInferenceClient
from tests.common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(autouse=True)
def clear_spans():
    custom_exporter.reset()
    yield

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="generic_hf_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        #wrapper_methods=HUGGING_FACE_METHODS,
    )

@pytest.mark.integration()
def test_huggingface_api_sample(setup):
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

    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")
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

    print("\n--- Captured span types ---")
    for span in spans:
        print(span.attributes.get("span.type", "NO_SPAN_TYPE"), "-", span.name)


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

@pytest.mark.integration()
@pytest.mark.asyncio
async def test_huggingface_api_async_sample(setup):
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

    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")
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

    print("\n--- Captured span types ---")
    for span in spans:
        print(span.attributes.get("span.type", "NO_SPAN_TYPE"), "-", span.name)

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

@pytest.mark.integration()
def test_huggingface_invalid_api_key(setup):
    client = InferenceClient(api_key="invalid_key_123")
    try:
        client.chat_completion(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "test"}],
        )
    except Exception as e:
        # Hugging Face returns a 401 HTTP error
        assert "401" in str(e) or "Unauthorized" in str(e)

    time.sleep(5)
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        print("SPAN:", span.name)
        for e in span.events:
            print(" EVENT:", e.name, e.attributes)

    for span in spans:
        if span.attributes.get("span.type") in ["inference", "inference.framework"]:
            events = [e for e in span.events if e.name == "data.output"]
            assert len(events) > 0
            assert span.status.status_code.value == 2
            error_code = events[0].attributes.get("error_code")
            assert error_code == "error"
            response = events[0].attributes.get("response")
            assert response is None or response == ""


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
