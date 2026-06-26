import json
import logging
import os
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from custom_litellm.llm import LiteLLMClient
from custom_litellm.prompt_loader import PromptLoader
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from tests.common.helpers import (
    find_spans_by_type,
    get_span_event_by_name,
    verify_inference_span,
)


@pytest.fixture(scope="function")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="litellm_gemini_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[]
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


logger = logging.getLogger(__name__)

template_config = {
    "name": "SentimentEval",
    "description": "Sentiment Classification task",
    "structured_input": ["input", "response"],
    "eval_prompt": "Please perform Sentiment Classification task. Given the sentence, assign a sentiment label from ['negative', 'positive', 'neutral']",
    "structure_output": {
        "label": {
            "description": "'negative' or 'positive' or 'neutral'"
        },
        "explanation": {
            "description": "Detailed analysis of the response's usefulness, covering relevance, clarity, accuracy, context, and actionability"
        }
    },
    "structure_error": "handle this -skip| static value also what type of error - refusal|error"

}


def evaluate_chat_completion(model):
    (template_name, response_model, sys_prompt) = PromptLoader.get_chat_prompt_template(template_config)
    try:
        logger.info(f"Attempting to call model: {model}")
        litellm_client = LiteLLMClient()
        response = litellm_client.get_completion(
            model=model,
            eval_prompt=sys_prompt,
            prompt="User : What is Coffee?\nAssistant : Don't ask silly questions like this.",
            response_format=response_model
        )
        time.sleep(10)
        logger.info(f"Response: {response}")

    except Exception as e:
        logger.info(f"An error occurred during call of {model}: {e}")
        assert False, f"Test failed due to exception: {e}"


@pytest.mark.unit()
def test_llm_gemini(setup):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key is None:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    model = "gemini/gemini-2.5-flash"
    evaluate_chat_completion(model)

    spans = setup.get_captured_spans()
    logger.info(f"Captured {len(spans)} spans")

    # Verify we have spans
    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        # Also check for inference.framework spans
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify each inference span
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.gemini",
            model_name="gemini-2.5-flash",
            model_type="model.llm.gemini-2.5-flash",
            check_metadata=True,
            check_input_output=True,
        )
    assert (
            len(inference_spans) == 1
    ), "Expected exactly one inference span for the LLM call"

    # Verify response_format (the Pydantic eval schema) is captured in data.input.
    data_input = get_span_event_by_name(inference_spans[0], "data.input")
    rf = data_input.attributes.get("response_format")
    assert rf is not None, "response_format missing from data.input span event"
    schema = json.loads(rf)
    properties = (
        schema.get("properties")
        or schema.get("json_schema", {}).get("schema", {}).get("properties", {})
    )
    assert "label" in properties, "Expected 'label' field in response_format schema"
    assert "explanation" in properties, "Expected 'explanation' field in response_format schema"


if __name__ == '__main__':
    logger.info("Starting pytest...")
    pytest.main(['-v', __file__])
    logger.info("Pytest finished.")
