import time
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="llama_index_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )

@pytest.mark.integration()
def test_llama_index_anthropic_sample(setup):
    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="Tell me a story"),
    ]
    llm = Anthropic(model="claude-3-5-sonnet-20240620")

    response = llm.chat(messages)

    print(response)
    time.sleep(5)
    spans = custom_exporter.get_captured_spans()

    llama_index_spans = [span for span in spans if span.name.startswith("langchain")]
    for span in llama_index_spans:
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

