# Test multiple chains with OpenAI APIs in between
import pytest
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from monocle_apptrace.instrumentation.common.instrumentor import (
    setup_monocle_telemetry,
    start_trace,
    stop_trace,
)
from openai import OpenAI
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from common.chain_exec import setup_simple_chain

custom_exporter = CustomConsoleSpanExporter()


@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="langchain_app_1",
        span_processors=[SimpleSpanProcessor(custom_exporter)],
        wrapper_methods=[],
    )


@pytest.fixture(autouse=True)
def pre_test():
    # clear old spans
    custom_exporter.reset()


# Test multiple chains with OpenAI APIs in between. Verify each has it's workflow and inference spans
@pytest.mark.integration()
def test_langchain_with_openai(setup):
    chain1 = setup_simple_chain()
    chain2 = setup_simple_chain()
    openai = OpenAI()

    chain1.invoke("What is an americano?")
    verify_spans(
        expected_langchain_inference_spans=1,
        expected_openai_inference_spans=0,
        exptected_workflow_spans=1,
    )
    custom_exporter.reset()

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an latte?"},
        ],
    )
    verify_spans(
        expected_langchain_inference_spans=0,
        expected_openai_inference_spans=1,
        exptected_workflow_spans=1,
    )
    custom_exporter.reset()

    chain2.invoke("What is an coffee?")
    verify_spans(
        expected_langchain_inference_spans=1,
        expected_openai_inference_spans=0,
        exptected_workflow_spans=1,
    )
    custom_exporter.reset()


# Test multiple chains with OpenAI APIs in between in a single trace Verify there only one workflow and all inference spans
@pytest.mark.integration()
def test_langchain_with_openai_single_trace(setup):
    chain1 = setup_simple_chain()
    chain2 = setup_simple_chain()
    openai = OpenAI()

    token = start_trace()
    chain1.invoke("What is an americano?")

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an latte?"},
        ],
    )
    chain2.invoke("What is an coffee?")
    stop_trace(token)
    verify_spans(
        expected_langchain_inference_spans=2,
        expected_openai_inference_spans=1,
        exptected_workflow_spans=1,
    )


def verify_spans(
    expected_langchain_inference_spans: int,
    expected_openai_inference_spans: int,
    exptected_workflow_spans: int,
):
    spans = custom_exporter.get_captured_spans()
    workflow_spans = 0
    langchain_inference_spans = 0
    openai_inference_spans = 0
    trace_id = -1
    trace_id = spans[0].context.trace_id
    for span in spans:
        span_attributes = span.attributes
        if trace_id == -1:
            trace_id = span.context.trace_id
        else:
            assert trace_id == span.context.trace_id
        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference"
            or span_attributes["span.type"] == "inference.framework"
        ):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            if span.name.lower().startswith("langchain"):
                langchain_inference_spans = langchain_inference_spans + 1
            elif span.name.lower().startswith("openai"):
                openai_inference_spans = openai_inference_spans + 1
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "workflow"
        ):
            workflow_spans = workflow_spans + 1

    assert expected_langchain_inference_spans == langchain_inference_spans
    assert expected_openai_inference_spans == openai_inference_spans
    assert exptected_workflow_spans == workflow_spans
    custom_exporter.reset()
