import logging
import os
import time

import pytest
import litellm
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from tests.common.helpers import (
    find_spans_by_type,
)


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry instrumentation for LiteLLM Azure OpenAI streaming tests."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter)
    ]
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="litellm_azure_stream_app_1",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


logger = logging.getLogger(__name__)


@pytest.mark.skipif(
    not (os.getenv("AZURE_API_KEY") and os.getenv("AZURE_API_BASE") and os.getenv("AZURE_API_VERSION")),
    reason="Azure credentials not configured"
)
def test_llm_azure_openai_stream(setup):
    """Test streaming LiteLLM Azure OpenAI completion with monocle instrumentation."""
    azure_api_key = os.getenv("AZURE_API_KEY")
    azure_api_base = os.getenv("AZURE_API_BASE")
    azure_api_version = os.getenv("AZURE_API_VERSION")

    try:
        logger.info("Attempting to call model: azure/gpt-4o-mini with streaming")

        # Prepare messages
        sys_prompt = "You are a helpful assistant."
        user_prompt = "What is the capital of France? Answer in one sentence."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Stream completion using LiteLLM with Azure OpenAI
        response_text = ""
        logger.info(f"Sending streaming request with messages: {messages}")

        stream_response = litellm.completion(
            model="azure/gpt-4o-mini",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            api_key=azure_api_key,
            api_base=azure_api_base,
            api_version=azure_api_version,
        )

        # Collect streaming chunks
        for chunk in stream_response:
            if chunk.choices and len(chunk.choices) > 0:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    response_text += delta_content
                    logger.debug(f"Chunk received: {delta_content}")

        logger.info(f"Full streamed response: {response_text}")
        assert response_text, "Expected to receive streamed content"
        assert len(response_text) > 10, "Expected substantive streamed response"

    except Exception as e:
        logger.error(f"An error occurred during streaming call: {e}")
        assert False, f"Test failed due to exception: {e}"

    time.sleep(2)  # Allow spans to be processed
    spans = setup.get_captured_spans()
    logger.info(f"Captured {len(spans)} spans")

    # Verify we have spans
    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        # Also check for inference.framework spans
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify we have streaming Azure OpenAI inference span
    verified = False
    for span in inference_spans:
        span_type = span.attributes.get("entity.1.type", "")
        logger.info(f"Checking span type: {span_type}")

        if "azure_openai" in span_type.lower():
            logger.info(f"Found Azure OpenAI inference span: {span.name}")

            # Verify basic span attributes
            assert span.attributes.get("span.type") in ["inference", "inference.framework"], \
                "Expected inference span type"
            assert span.attributes.get("entity.2.name") == "gpt-4o-mini", "Expected gpt-4o-mini model"
            assert span.attributes.get("entity.2.type") == "model.llm.gpt-4o-mini", "Expected model type"

            # Verify response contains the streamed content
            has_output = False
            has_tokens = False
            for event in span.events:
                if event.name == "data.output":
                    output = event.attributes.get("response", "")
                    logger.info(f"Output event found: {output[:100] if output else 'empty'}...")
                    if output:
                        has_output = True
                        assert "paris" in output.lower() or "france" in output.lower() or len(output) > 20, \
                            f"Expected meaningful output, got: {output[:100]}"
                if event.name == "metadata":
                    completion_tokens = event.attributes.get("completion_tokens", 0)
                    prompt_tokens = event.attributes.get("prompt_tokens", 0)
                    total_tokens = event.attributes.get("total_tokens", 0)
                    logger.info(
                        f"Metadata tokens: completion={completion_tokens}, prompt={prompt_tokens}, total={total_tokens}"
                    )
                    if total_tokens or completion_tokens or prompt_tokens:
                        has_tokens = True

            if has_output and has_tokens:
                verified = True
                logger.info("Azure OpenAI streaming response and token usage captured and verified")
                break

    assert verified, "Expected to find inference span with streamed output and token metadata"


if __name__ == '__main__':
    logger.info("Starting pytest...")
    pytest.main(['-v', __file__])
    logger.info("Pytest finished.")
