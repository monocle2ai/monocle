import time
import os
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
                workflow_name="langchain_app_1",
                span_processors=[BatchSpanProcessor(custom_exporter)],
                wrapper_methods=[])

@pytest.mark.integration()
def test_langchain_anthropic_sample(setup):
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    ai_answer = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    time.sleep(5)
    print(ai_answer)

    spans = custom_exporter.get_captured_spans()

    langchain_spans = [span for span in spans if span.name.startswith("langchain")]

    for span in langchain_spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes

            assert span_attributes["entity.1.type"] == "inference.anthropic"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "claude-3-5-sonnet-20240620"
            assert span_attributes["entity.2.type"] == "model.llm.claude-3-5-sonnet-20240620"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if not span.parent and span.name == "langchain.workflow":
            assert span_attributes["entity.1.name"] == "langchain_app_1"
            assert span_attributes["entity.1.type"] == "workflow.langchain"