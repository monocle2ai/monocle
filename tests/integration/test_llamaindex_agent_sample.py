import time
from tests.common.custom_exporter import CustomConsoleSpanExporter
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from llama_index.core.agent import ReActAgent
import pytest
custom_exporter = CustomConsoleSpanExporter()
@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="llama_index_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )

def add(x: int, y: int) -> int:
    """Useful function to add two numbers."""
    return x + y


def multiply(x: int, y: int) -> int:
    """Useful function to multiply two numbers."""
    return x * y


tools = [
    FunctionTool.from_defaults(add),
    FunctionTool.from_defaults(multiply),
]

def test_llamaindex_agent(setup):
    llm=OpenAI(temperature=0.1, model="gpt-4")

    agent = ReActAgent.from_tools(
        tools=tools, llm=llm, memory=None, verbose=True
    )

    ret = agent.chat("What is (2123 + 2321) * 312?")
    time.sleep(5)
    print(ret.response)

    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.azure_oai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "agent":
            # Assertions for all inference attributes
            assert span_attributes["entity.2.name"] == "ReActAgent"
            assert span_attributes["entity.2.type"] == "Agent.oai"
            assert span_attributes["entity.2.tools"] == ('add','multiply',)


