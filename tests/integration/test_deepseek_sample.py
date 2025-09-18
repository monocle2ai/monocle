import os
import time
import asyncio
from openai import AsyncOpenAI  # Use the async client
from xmlrpc import client
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.utils import logger
from monocle_apptrace.instrumentation.common.instrumentor import set_context_properties, setup_monocle_telemetry
from openai import OpenAI, OpenAIError

from monocle_apptrace.instrumentation.metamodel.openai import _helper
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
    # Setup Monocle telemetry
    setup_monocle_telemetry(
        workflow_name="generic_deepseek_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[],
    )

@pytest.mark.integration()
def test_deepseek_api_sample(setup):
    #set_context_properties({"entity.1.type": "workflow.deepseek"})

    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for coffee related questions"},
            {"role": "user", "content": "What is an americano?"},
        ],
    )

    time.sleep(5)

    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")

    # Verify spans exist
    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify inference spans
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.deepseek",
            model_name="deepseek-chat",
            model_type="model.llm.deepseek-chat",
            check_metadata=True,
            check_input_output=True,
        )

    # Keep only spans from wrapper.py or test.py depending on what you want
    inference_spans = [s for s in inference_spans if s.attributes.get("span.type", "") == "inference"]
    
    assert ( len(inference_spans) == 1 ), "Expected exactly one inference span for the LLM call"

    # Validate span events
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
    assert workflow_span.attributes["entity.1.name"] == "generic_deepseek_1"
    assert workflow_span.attributes["workflow.name"] == "generic_deepseek_1"
    #using openai sdk for deepseek
    assert workflow_span.attributes["entity.1.type"] == "workflow.openai"

@pytest.mark.integration()
@pytest.mark.asyncio
async def test_deepseek_api_sample_async(setup):
    #set_context_properties({"entity.1.type": "workflow.deepseek"})

    client = AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    print(client._client.base_url)
    print(_helper.get_inference_type(client))

    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for coffee related questions"},
            {"role": "user", "content": "What is an americano?"},
        ],
    )

    # Wait briefly to ensure spans are processed
    await asyncio.sleep(5)

    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")

    # Verify spans exist
    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify inference spans
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.deepseek",
            model_name="deepseek-chat",
            model_type="model.llm.deepseek-chat",
            check_metadata=True,
            check_input_output=True,
        )

    # Keep only spans from wrapper.py or test.py depending on what you want
    inference_spans = [s for s in inference_spans if s.attributes.get("span.type", "") == "inference"]

    assert len(inference_spans) == 1, "Expected exactly one inference span for the LLM call"

    # Validate span events
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
    assert workflow_span.attributes["entity.1.name"] == "generic_deepseek_1"
    assert workflow_span.attributes["workflow.name"] == "generic_deepseek_1"
    #using openai sdk for deepseek
    assert workflow_span.attributes["entity.1.type"] == "workflow.openai"


@pytest.mark.integration()
def test_deepseek_invalid_api_key(setup):
    try:
        client = OpenAI(api_key="invalid_key_123", base_url="https://api.deepseek.com")
        client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "test"}],
        )
    except OpenAIError as e:
        logger.error("Authentication error: %s", str(e))

    time.sleep(5)
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        if span.attributes.get("span.type") in ["inference", "inference.framework"]:
            events = [e for e in span.events if e.name == "data.output"]
            assert len(events) > 0
            assert span.status.status_code.value == 2  # ERROR
            # DeepSeek returns 'invalid_request_error' instead of 'invalid_api_key'
            assert events[0].attributes["error_code"] in [
                "invalid_request_error",
                "invalid_api_key",  # fallback if they align later
            ]
            assert "error code: 401" in events[0].attributes.get("response", "").lower()



if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
