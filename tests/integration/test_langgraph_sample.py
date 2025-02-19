from langchain.tools.base import BaseTool
from langchain_openai import ChatOpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from tests.common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import Tool
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

coffee_menu = {
    "espresso": "A strong and bold coffee shot.",
    "latte": "A smooth coffee with steamed milk.",
    "cappuccino": "A rich coffee with frothy milk foam.",
    "americano": "Espresso with added hot water for a milder taste.",
    "mocha": "A chocolate-flavored coffee with whipped cream."
}

def get_coffee_menu(_):
    """Returns the available coffee menu."""
    return "\n".join([f"{k}: {v}" for k, v in coffee_menu.items()])

def order_coffee(order: str):
    """Processes a coffee order."""
    order = order.lower()
    if order in coffee_menu:
        return f"Your {order} is being prepared. Enjoy your coffee!"
    else:
        return "Sorry, we don’t have that coffee option. Please choose from the menu."

def get_coffee_details(coffee_name: str):
    """Provides details about a specific coffee."""
    coffee_name = coffee_name.lower()
    return coffee_menu.get(coffee_name, "Sorry, we don’t have details for that coffee.")

# Define tools for the chatbot
tools = [
    Tool(
        name="GetCoffeeMenu",
        func=get_coffee_menu,
        description="Returns a list of available coffee options."
    ),
    Tool(
        name="OrderCoffee",
        func=order_coffee,
        description="Orders a coffee from the menu. Provide the coffee name."
    ),
    Tool(
        name="GetCoffeeDetails",
        func=get_coffee_details,
        description="Provides details about a specific coffee. Input the coffee name."
    )
]

# Set up the Chat Model
llm = ChatOpenAI(model_name="gpt-4", temperature=0)


@pytest.mark.integration()
def test_langgraph_chat_sample(setup):

    agent_executor = create_react_agent(llm, tools)

    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}

    question = "order 1 espresso"
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
            assert span_attributes["entity.2.name"] == "gpt-4"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4"

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
            assert span_attributes["entity.2.tools"]  == ('GetCoffeeMenu', 'OrderCoffee', 'GetCoffeeDetails')

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes


