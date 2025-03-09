import os
import time
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from openai import OpenAI
custom_exporter = CustomConsoleSpanExporter()

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
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
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