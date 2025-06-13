import os
import time
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.utils import logger
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from openai import OpenAI, OpenAIError
custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(autouse=True)
def clear_spans():
    """Clear spans before each test"""
    custom_exporter.reset()
    yield

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
                workflow_name="langchain_app_1",
                span_processors=[BatchSpanProcessor(custom_exporter)],
                wrapper_methods=[])

@pytest.mark.integration()
def test_openai_api_sample(setup):
    openai = OpenAI()
    response = openai.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": "You are a helpful assistant to answer coffee related questions"},
        {"role": "user", "content": "What is an americano?"}
      ]
      )
    time.sleep(5)
    print(response)
    print(response.choices[0].message.content)

    spans = custom_exporter.get_captured_spans()
    found_workflow_span = False
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4o-mini"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4o-mini"

            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
        
        if "span.type" in span_attributes and span_attributes["span.type"] == "workflow":
            found_workflow_span = True
    assert found_workflow_span


@pytest.mark.integration()
def test_openai_invalid_api_key(setup):
    try:
        client = OpenAI(api_key="invalid_key_123")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )
    except OpenAIError as e:
        logger.error("Authentication error: %s", str(e))

    time.sleep(5)
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        if span.attributes.get("span.type") == "inference" or span.attributes.get("span.type") == "inference.framework":
            events = [e for e in span.events if e.name == "data.output"]
            assert len(events) > 0
            assert events[0].attributes["status"] == "error"
            assert events[0].attributes["status_code"] == "invalid_api_key"
            assert "error code: 401" in events[0].attributes.get("response", "").lower()
