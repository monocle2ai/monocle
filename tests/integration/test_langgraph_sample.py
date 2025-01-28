from langchain.tools.base import BaseTool
from langchain_openai import ChatOpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from tests.common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, List, Optional, Type, Union
from langgraph.prebuilt import create_react_agent
import pytest
import os
import logging
logger = logging.getLogger(__name__)

import time
custom_exporter = CustomConsoleSpanExporter()


@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
                workflow_name="langchain_app_1",
                span_processors=[BatchSpanProcessor(custom_exporter)],
                wrapper_methods=[])


class FAQInput(BaseModel):
    """Input for the Coffee tool."""

    query: str = Field(description="enter the query")


class FAQTool(BaseTool):
    name: str = "faq_tool"
    description : str = (
        "This tool answers frequently asked questions (FAQs). "
        "It provides pre-defined answers to common questions."
    )

    # Define the FAQs
    faq_data : Dict[str, str] = {
        "What is React?": "React is a JavaScript library for building user interfaces, maintained by Facebook.",
        "What is Python?": "Python is a versatile programming language known for its readability and wide usage in various domains.",
        "What is LangChain?": "LangChain is a framework designed to help develop applications powered by language models.",
        "What is AI?": "AI, or Artificial Intelligence, refers to the simulation of human intelligence in machines.",
    }

    def _run(self, query: str):
        # Handle synchronous responses
        response = self.faq_data.get(query, "Sorry, I do not have an answer to that question.")
        return response

    async def _arun(self, query: str):
        # Handle asynchronous responses
        response = self.faq_data.get(query, "Sorry, I do not have an answer to that question.")
        return response

@pytest.mark.integration()
def test_langgraph_chat_sample(setup):


    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    tools = [FAQTool()]

    agent_executor = create_react_agent(llm, tools )

    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}

    question = "What is React?"
    chunk = agent_executor.invoke({"messages": [HumanMessage(content=question)]})
    if len(chunk["messages"]) > 2:
        tool_msg: ToolMessage = chunk["messages"][2]
        print(tool_msg.content)
    else:

        print("Sorry, I can't help you with " + question)
    time.sleep(5)
    spans = custom_exporter.get_captured_spans()

    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.azure_oai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-3.5-turbo-0125"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-3.5-turbo-0125"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "agent":
            assert "entity.2.type" in span_attributes
            assert "entity.2.name" in span_attributes
            assert "entity.2.tools" in span_attributes
            assert span_attributes["entity.2.name"] == "LangGraph"
            assert span_attributes["entity.2.type"] == "agent.oai"
            assert span_attributes["entity.2.tools"]  == ('faq_tool',)

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes


