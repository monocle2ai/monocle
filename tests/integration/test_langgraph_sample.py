from langchain.tools.base import BaseTool
from langchain_openai import ChatOpenAI
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from tests.common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import Tool
# from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, List, Optional, Type, Union
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import pytest
import os
import logging
import asyncio

logger = logging.getLogger(__name__)

import time

memory_exporter = CustomConsoleSpanExporter()
span_processors = [SimpleSpanProcessor(memory_exporter)]
os.environ["MONOCLE_EXPORTER"] = "okahu,file"
# os.environ["MONOCLE_EXPORTER"] = "file"
memory_exporter = InMemorySpanExporter()
span_processors=[SimpleSpanProcessor(memory_exporter)]

# @pytest.fixture(scope="module")
def setup():
    memory_exporter.reset()
    setup_monocle_telemetry(
        workflow_name="langchain_agent_1",
        span_processors=[
            # SimpleSpanProcessor(memory_exporter),
            # BatchSpanProcessor(FileSpanExporter())
        ],
    # memory_exporter.clear()
    # setup_monocle_telemetry(
    #             workflow_name="langchain_agent_1",
    #             span_processors=[SimpleSpanProcessor(memory_exporter)]
    #             )
    memory_exporter.clear()
    setup_monocle_telemetry(
                workflow_name="langchain_agent_1",
                span_processors=[SimpleSpanProcessor(memory_exporter)]
                )

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


# Set up MCP client for monocle repo
async def get_mcp_tools():
    """Get tools from the monocle MCP server."""
    client = MultiServerMCPClient(
        {
            "monocle": {
                "url": "http://localhost:8000/search_text/mcp/",
                "transport": "streamable_http",
            },
            "okahu": {
                "url": "http://localhost:8000/okahu/mcp/",
                "transport": "streamable_http",
            },
        }
    )
    # get list of all tools from the MCP server and their descriptions
    tools = await client.get_tools()
    return tools


# Set up the Chat Model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# @pytest.mark.integration()
# @pytest.mark.asyncio
async def test_langgraph_chat_sample():

    async def run_test():
        # Get tools from MCP server
        tools = await get_mcp_tools()
        local_tools = []
        full_tools = tools + local_tools

        agent_executor = create_react_agent(llm, full_tools)

        # Use the agent

        question = "Search monocle for 'def extract_messages'"
        chunk = await agent_executor.ainvoke(
            {"messages": [HumanMessage(content=question)]}
        )
        # agent invcation => question, list of tools
        # openai llm => question, list of tools
        # if a tool is returned, it will be executed
        
        
        # direct mcp
        if len(chunk["messages"]) > 2:
            tool_msg: ToolMessage = chunk["messages"][2]
            # print(tool_msg.content)
        else:
            print("Sorry, I can't help you with " + question)
    question = "order an espresso"
    chunk = agent_executor.invoke({"messages": [HumanMessage(content=question)]})
    if len(chunk["messages"]) > 2:
        tool_msg: ToolMessage = chunk["messages"][2]
        print(tool_msg.content)
    else:

        print("Sorry, I can't help you with " + question)
    time.sleep(5)
    spans = memory_exporter.get_finished_spans()

    found_inference = found_agent = found_tool = False

    found_inference = found_agent = found_tool = False

        return chunk
    
    async def run_test_okahu():
        tools = await get_mcp_tools()
        agent_executor = create_react_agent(llm, tools)

        # question = """Get the app that starts with chatbot-coffee-vercel.
        #     I dont remember the exact name, but I know it starts with chatbot-coffee-vercel.
        #     check the SLA of the app and download its last 5 traces as well.
        #     Check what is the input prompt in the trace and tell me if somebody is trying to hack into my chatbot.
            
        # """
        
        question = """Get the app that starts with chatbot-coffee-vercel.
            I dont remember the exact name, but I know it starts with chatbot-coffee-vercel.
            Dont use the traces or spans api.
            Check what is the input prompt in the trace and tell me if somebody is trying to hack into my chatbot.
            Use the prompts api to get the input prompt for the app.
        """
        chunk = await agent_executor.ainvoke(
            {"messages": [HumanMessage(content=question)]}
        )
        # openai llm => question, list of tools
        # 
        if len(chunk["messages"]) > 2:
            tool_msg: ToolMessage = chunk["messages"][2]
            # print(tool_msg.content)
        else:
            print("Sorry, I can't help you with " + question)

        return chunk
        
    # Run the async test
    chunk = await run_test_okahu()

    # print("Chunk received:", chunk)

    # await asyncio.sleep(1)
    # spans = memory_exporter.get_finished_spans()
    
    # print("Spans received:", spans)

    # found_inference = found_agent = found_tool = False

    # for span in spans:
    #     span_attributes = span.attributes

    #     if "span.type" in span_attributes and (
    #         span_attributes["span.type"] == "inference"
    #         or span_attributes["span.type"] == "inference.framework"
    #     ):
    #         # Assertions for all inference attributes
    #         assert span_attributes["entity.1.type"] == "inference.openai"
    #         assert "entity.1.provider_name" in span_attributes
    #         assert "entity.1.inference_endpoint" in span_attributes
    #         assert span_attributes["entity.2.name"] == "gpt-4"
    #         assert span_attributes["entity.2.type"] == "model.llm.gpt-4"

    #         # Assertions for metadata
    #         span_input, span_output, span_metadata = span.events
    #         assert "completion_tokens" in span_metadata.attributes
    #         assert "prompt_tokens" in span_metadata.attributes
    #         assert "total_tokens" in span_metadata.attributes
    #         found_inference = True

    #     if (
    #         "span.type" in span_attributes
    #         and span_attributes["span.type"] == "agentic.invocation"
    #     ):
    #         assert "entity.1.type" in span_attributes
    #         assert "entity.1.name" in span_attributes
    #         assert span_attributes["entity.1.name"] == "LangGraph"
    #         assert span_attributes["entity.1.type"] == "agent.langgraph"
    #         found_agent = True

    #     if (
    #         "span.type" in span_attributes
    #         and span_attributes["span.type"] == "agentic.tool"
    #     ):
    #         assert "entity.1.type" in span_attributes
    #         assert "entity.1.name" in span_attributes
    #         # Updated assertion to match MCP tool names (will vary based on actual MCP server tools)
    #         assert span_attributes["entity.1.type"] == "tool.langgraph"
    #         found_tool = True

    # assert found_inference, "Inference span not found"
    # assert found_agent, "Agent span not found"
    # assert found_tool, "Tool span not found"
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

if __name__ == "__main__":
    setup()
    asyncio.run(test_langgraph_chat_sample())
    print("Test completed.")
    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
