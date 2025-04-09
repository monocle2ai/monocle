import os
import pytest
import time
import anthropic
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="anthropic_app_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[
        ])

@pytest.mark.integration()
def test_anthropic_metamodel_sample(setup):
    client = anthropic.Anthropic()

    # Send a prompt to Claude
    response = client.messages.create(
        model="claude-3-opus-20240229",  # You can use claude-3-haiku, claude-3-sonnet, etc.
        max_tokens=512,
        temperature=0.7,
        messages=[
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
    )

    # Print the response
    print("Claude's response:\n")
    print(response.content[0].text)

    time.sleep(5)
    spans = custom_exporter.get_captured_spans()

    for span in spans:
        span_attributes = span.attributes
        if span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.generic"
            assert span_attributes["entity.1.provider_name"] == "api.anthropic.com"
            assert span_attributes["entity.1.inference_endpoint"] == "https://api.anthropic.com"
            assert span_attributes["entity.2.name"] == "claude-3-opus-20240229"
            assert span_attributes["entity.2.type"] == "model.llm.claude-3-opus-20240229"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
