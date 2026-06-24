import logging
import os
import time

import pytest
import litellm
from common.custom_exporter import CustomConsoleSpanExporter
from custom_litellm.prompt_loader import PromptLoader
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from tests.common.helpers import (
    find_spans_by_type,
)


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry instrumentation for LiteLLM streaming tests."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter)
    ]
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="litellm_stream_app_1",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


logger = logging.getLogger(__name__)


def test_llm_openai_stream(setup):
    """Test streaming LiteLLM OpenAI completion with monocle instrumentation."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    try:
        logger.info("Attempting to call model: gpt-4o-mini with streaming")
        
        # Prepare messages
        sys_prompt = "You are a helpful assistant."
        user_prompt = "What is the capital of France? Answer in one sentence."
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Stream completion using LiteLLM directly
        response_text = ""
        logger.info(f"Sending streaming request with messages: {messages}")
        
        stream_response = litellm.completion(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
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

    # Verify we have streaming inference span
    verified = False
    for span in inference_spans:
        span_type = span.attributes.get("entity.1.type", "")
        logger.info(f"Checking span type: {span_type}")
        
        if "openai" in span_type.lower():
            logger.info(f"Found OpenAI inference span: {span.name}")
            
            # Verify basic span attributes
            assert span.attributes.get("span.type") in ["inference", "inference.framework"], "Expected inference span type"
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
                        # Verify output looks like a response
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
                logger.info("Streaming response and token usage captured and verified")
                break
    
    assert verified, "Expected to find inference span with streamed output and token metadata"


if __name__ == '__main__':
    logger.info("Starting pytest...")
    pytest.main(['-v', __file__])
    logger.info("Pytest finished.")
