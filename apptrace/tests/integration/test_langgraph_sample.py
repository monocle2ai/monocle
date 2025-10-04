import logging
import time

import pytest
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def setup():
    memory_exporter = InMemorySpanExporter()
    span_processors = [SimpleSpanProcessor(memory_exporter)]
    try:
        instrumentor = setup_monocle_telemetry(
                    workflow_name="langchain_agent_1",
                    span_processors=[SimpleSpanProcessor(memory_exporter)]
                    )
        yield memory_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

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


def test_langgraph_chat_sample(setup):

    agent_executor = create_react_agent(llm, tools)

    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}

    question = "order an espresso"
    chunk = agent_executor.invoke({"messages": [HumanMessage(content=question)]})
    if len(chunk["messages"]) > 2:
        tool_msg: ToolMessage = chunk["messages"][2]
        logger.info(tool_msg.content)
    else:

        logger.info("Sorry, I can't help you with " + question)
    time.sleep(5)
    spans = setup.get_finished_spans()

    found_inference = found_agent = found_tool = False

    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
            found_inference = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.invocation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.name"] == "LangGraph"
            assert span_attributes["entity.1.type"] == "agent.langgraph"
            found_agent = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.tool.invocation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.name"] == "OrderCoffee"
            assert span_attributes["entity.1.type"] == "tool.langgraph"
            found_tool = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
