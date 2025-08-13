import logging
import os
import time

import pytest
from custom_litellm.prompt_loader import PromptLoader
from custom_litellm.llm import LiteLLMClient
from common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from tests.common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)
custom_exporter = CustomConsoleSpanExporter()
@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="litellm_app_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )


@pytest.fixture(autouse=True)
def clear_spans():
    """Clear spans before each test"""
    custom_exporter.reset()
    yield

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
        response =  litellm_client.get_completion(
            model=model,
            eval_prompt=sys_prompt,
            prompt="User : What is Coffee?\nAssistant : Don't ask silly questions like this.",
            response_format= response_model
        )
        time.sleep(10)
        logger.info(f"Response: {response}")
        print("Response:" ,response)


    except Exception as e:
        logger.info(f"An error occurred during call of {model}: {e}")
        assert False, f"Test failed due to exception: {e}"


@pytest.mark.integration()
def test_llm_openai(setup):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    model = "gpt-4o-mini"
    evaluate_chat_completion(model)

    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")

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
            entity_type="inference.openai",
            model_name="gpt-4o-mini",
            model_type="model.llm.gpt-4o-mini",
            check_metadata=True,
            check_input_output=True,
        )
    assert (
            len(inference_spans) == 1
    ), "Expected exactly one inference span for the LLM call"



@pytest.mark.unit()
def test_llm_azure_openai(setup):
    azure_api_key = os.getenv("AZURE_API_KEY")
    azure_api_base = os.getenv("AZURE_API_BASE")
    azure_api_version = os.getenv("AZURE_API_VERSION")

    if azure_api_key is None or azure_api_version is None or azure_api_base is None:
        raise ValueError("AZURE_API_KEY, AZURE_API_VERSION, AZURE_API_BASE environment variables must be set.")
    model = "azure/gpt-4o-mini"
    evaluate_chat_completion(model)

    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")

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
            entity_type="inference.azure_openai",
            model_name="gpt-4o-mini",
            model_type="model.llm.gpt-4o-mini",
            check_metadata=True,
            check_input_output=True,
        )
    assert (
            len(inference_spans) == 1
    ), "Expected exactly one inference span for the LLM call"

if __name__ == '__main__':
    logger.info("Starting pytest...")
    pytest.main(['-v', __file__])
    logger.info("Pytest finished.")