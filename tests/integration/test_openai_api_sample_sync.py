import os
import time
import unittest
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
        wrapper_methods=[],
    )


@pytest.mark.integration()
def test_openai_api_sample(setup):
    openai = OpenAI()
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )
    time.sleep(5)
    print(response)
    print(response.choices[0].message.content)

    spans = custom_exporter.get_captured_spans()
    found_workflow_span = False
    inference_span = None
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference"
            or span_attributes["span.type"] == "inference.framework"
        ):
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
            inference_span = span


        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "workflow"
        ):
            found_workflow_span = True
    assert found_workflow_span
    assert inference_span is not None


@pytest.mark.integration()
def test_openai_api_sample_stream(setup):
    openai = OpenAI()
    stream = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
        stream=True,
    )

    # Collect the streamed response
    collected_chunks = []
    collected_messages = []

    for chunk in stream:
        collected_chunks.append(chunk)
        if chunk.choices[0].delta.content is not None:
            collected_messages.append(chunk.choices[0].delta.content)

    full_response = "".join(collected_messages)
    print("Streamed response:", full_response)

    # Wait for spans to be processed
    time.sleep(5)

    spans = custom_exporter.get_captured_spans()
    found_workflow_span = False
    inference_span = None
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference"
            or span_attributes["span.type"] == "inference.framework"
        ):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4o-mini"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4o-mini"
            inference_span = span

            # TODO: Uncomment this when metadata is available
            # span_input, span_output, span_metadata = span.events
            # assert "completion_tokens" in span_metadata.attributes
            # assert "prompt_tokens" in span_metadata.attributes
            # assert "total_tokens" in span_metadata.attributes

        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "workflow"
        ):
            found_workflow_span = True
    assert found_workflow_span
    assert inference_span is not None

    # Assert we got a valid response
    assert len(collected_chunks) > 0
    assert len(full_response) > 0


def run_test():
    """Run the test directly without pytest"""
    # Call the setup function directly
    setup_monocle_telemetry(
        workflow_name="langchain_app_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[],
    )

    # Call the test functions directly
    print("Running non-streaming test:")
    test_openai_api_sample(None)

    # Clear the exporter before the second test
    custom_exporter.reset()

    print("\nRunning streaming test:")
    test_openai_api_sample_stream(None)

    print("All tests completed successfully")


if __name__ == "__main__":
    run_test()
