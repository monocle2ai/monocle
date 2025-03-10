import os
import bs4
import pytest
import asyncio
from common.chain_exec import setup_chain
from openai import AsyncOpenAI
from common.custom_exporter import CustomConsoleSpanExporter
from langchain_chroma import Chroma
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

custom_exporter = CustomConsoleSpanExporter()
@pytest.fixture(autouse=True)
def pre_test():
    # clear old spans
    custom_exporter.reset()

@pytest.fixture(scope="session")
def setup():
    setup_monocle_telemetry(
                workflow_name="langchain_app_1"
                ,span_processors=[SimpleSpanProcessor(custom_exporter)],
    )

@pytest.mark.integration()
def test_async_langchain_invoke(setup):
    chain = setup_chain()
    result= asyncio.run(chain.ainvoke("What is Task Decomposition?"))
    print(result)
    verify_inferece_span()

@pytest.mark.integration()
def test_async_openAI(setup):
    openai = AsyncOpenAI()
    response = asyncio.run( openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant to answer coffee related questions"},
            {"role": "user", "content": "What is an americano?"}
        ]
    ))
    print(response)
    verify_inferece_span()

def verify_inferece_span():
    spans = custom_exporter.get_captured_spans()
    found_inferece_span:bool = False
    for span in spans:
        span_attributes = span.attributes
        span_events = span.events
        if span_attributes.get("span.type") == "inference":
            found_inferece_span = True
            break
    assert found_inferece_span
