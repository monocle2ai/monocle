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


@pytest.fixture(scope="function")
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


# ---------------------------------------------------------------------------
# span.subtype tests
# ---------------------------------------------------------------------------

HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
HF_MODEL = os.getenv("HUGGING_FACE_MODEL", "openai/gpt-oss-120b")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
    },
}


@pytest.mark.skipif(not HF_API_KEY, reason="HUGGING_FACE_API_KEY not set")
def test_hf_subtype_tool_call(setup):
    """Inference span gets span.subtype='tool_call' when the HF model calls a tool."""
    baseline_count = len(setup.get_captured_spans())
    client = InferenceClient(api_key=HF_API_KEY)
    
    client.chat_completion(
        model=HF_MODEL,
        messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        max_tokens=100,
    )
    
    time.sleep(2)
    spans = setup.get_captured_spans()[baseline_count:]
    assert spans, "No spans captured"

    inference_spans = [
        s for s in spans
        if s.attributes.get("span.type") == "inference"
    ]
    assert inference_spans, "No inference spans captured"

    tool_call_spans = [
        s for s in inference_spans
        if s.attributes.get("span.subtype") == "tool_call"
    ]
    assert tool_call_spans, (
        "Expected at least one inference span with span.subtype='tool_call'. "
        f"Got subtypes: {[s.attributes.get('span.subtype') for s in inference_spans]}"
    )

    span = tool_call_spans[0]
    assert span.attributes.get("entity.3.name") == "get_weather", (
        f"Expected entity.3.name='get_weather', got '{span.attributes.get('entity.3.name')}'"
    )
    assert span.attributes.get("entity.3.type") == "tool.huggingface", (
        f"Expected entity.3.type='tool.huggingface', got '{span.attributes.get('entity.3.type')}'"
    )
    logger.info("✓ HuggingFace: span.subtype='tool_call' and entity.3 verified")


@pytest.mark.skipif(not HF_API_KEY, reason="HUGGING_FACE_API_KEY not set")
def test_hf_subtype_turn_end(setup):
    """Inference span gets span.subtype='turn_end' when HF model responds without a tool call."""
    baseline_count = len(setup.get_captured_spans())
    client = InferenceClient(api_key=HF_API_KEY)
    
    client.chat_completion(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Say hello in one word."},
        ],
        max_tokens=10,
    )

    time.sleep(2)
    spans = setup.get_captured_spans()[baseline_count:]
    assert spans, "No spans captured"

    inference_spans = [
        s for s in spans
        if s.attributes.get("span.type") == "inference"
    ]
    assert inference_spans, "No inference spans captured"

    span = inference_spans[-1]
    assert "span.subtype" in span.attributes, "span.subtype attribute must be present"
    assert span.attributes.get("span.subtype") == "turn_end", (
        f"Expected span.subtype='turn_end', got '{span.attributes.get('span.subtype')}'"
    )
    logger.info("✓ HuggingFace: span.subtype='turn_end' verified")


@pytest.mark.skipif(not HF_API_KEY, reason="HUGGING_FACE_API_KEY not set")
def test_hf_entity3_absent_on_turn_end(setup):
    """entity.3.name should not be set when the HF model does not call a tool."""
    baseline_count = len(setup.get_captured_spans())
    client = InferenceClient(api_key=HF_API_KEY)
    
    client.chat_completion(
        model=HF_MODEL,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        max_tokens=20,
    )

    time.sleep(2)
    spans = setup.get_captured_spans()[baseline_count:]
    assert spans, "No spans captured"

    turn_end_spans = [
        s for s in spans
        if s.attributes.get("span.type") == "inference"
        and s.attributes.get("span.subtype") == "turn_end"
    ]
    assert turn_end_spans, "Expected a turn_end inference span"
    assert turn_end_spans[0].attributes.get("entity.3.name") is None, (
        "entity.3.name should not be set when there is no tool call"
    )
    logger.info("✓ entity.3.name correctly absent on turn_end inference span")


@pytest.mark.asyncio
@pytest.mark.skipif(not HF_API_KEY, reason="HUGGING_FACE_API_KEY not set")
async def test_hf_async_subtype_tool_call(setup):
    """Async: inference span gets span.subtype='tool_call' when HF model calls a tool."""
    baseline_count = len(setup.get_captured_spans())
    client = AsyncInferenceClient(api_key=HF_API_KEY)
    
    await client.chat_completion(
        model=HF_MODEL,
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        max_tokens=100,
    )

    time.sleep(2)
    spans = setup.get_captured_spans()[baseline_count:]
    assert spans, "No spans captured"

    inference_spans = [
        s for s in spans
        if s.attributes.get("span.type") == "inference"
    ]
    assert inference_spans, "No inference spans captured"

    tool_call_spans = [
        s for s in inference_spans
        if s.attributes.get("span.subtype") == "tool_call"
    ]
    assert tool_call_spans, (
        "Expected at least one inference span with span.subtype='tool_call'. "
        f"Got subtypes: {[s.attributes.get('span.subtype') for s in inference_spans]}"
    )

    span = tool_call_spans[0]
    assert span.attributes.get("entity.3.name") == "get_weather", (
        f"Expected entity.3.name='get_weather', got '{span.attributes.get('entity.3.name')}'"
    )
    logger.info("✓ HuggingFace async: span.subtype='tool_call' and entity.3.name verified")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
