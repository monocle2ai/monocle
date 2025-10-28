import asyncio
import logging
import time

import pytest
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def setup():
    instrumentor = None
    memory_exporter = InMemorySpanExporter()
    span_processors = [SimpleSpanProcessor(memory_exporter)]
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="llama_index_1",
            span_processors=span_processors
        )
        yield memory_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

# Define coffee menu
COFFEE_MENU = {
    "espresso": 2.5,
    "latte": 3.5,
    "cappuccino": 4.0,
    "americano": 3.0
}

# Define tools for the chatbot

def get_coffee_menu() -> str:
    """Return the available coffee menu."""
    menu_str = "\n".join([f"{item}: ${price:.2f}" for item, price in COFFEE_MENU.items()])
    return f"Available coffee options:\n{menu_str}"

coffee_menu_tool = FunctionTool.from_defaults(
    fn=get_coffee_menu,
    name="get_coffee_menu",
    description="Provides a list of available coffee options with prices."
)

def place_order(coffee_type: str, quantity: int) -> str:
    """Places an order for coffee."""
    if coffee_type.lower() not in COFFEE_MENU:
        return f"Sorry, {coffee_type} is not available. Please choose from the menu."
    total_cost = COFFEE_MENU[coffee_type.lower()] * quantity
    return f"Your order for {quantity} {coffee_type}(s) is confirmed. Total cost: ${total_cost:.2f}"

order_tool = FunctionTool.from_defaults(
    fn=place_order,
    name="place_order",
    description="Takes a coffee order and provides the total cost."
)

# Initialize LlamaIndex ReAct agent
llm = OpenAI(model="gpt-4")
agent = ReActAgent(name="ReActAgent",tools=[coffee_menu_tool, order_tool], llm=llm)

def test_llamaindex_agent(setup):
    logger.info("Welcome to the Coffee Bot! ")
    user_input = "Please order 3 espresso coffees"
    
    async def run_agent():
        response = await agent.run(user_input)
        return response
    
    response = asyncio.run(run_agent())
    time.sleep(5)
    logger.info(f"Bot: {response}")

    spans = setup.get_finished_spans()
    found_inference_span = found_agent_span = found_tool_span = False
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.modelapi"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4"
            found_inference_span = True

            # Assertions for metadata
            # span_input, span_output, span_metadata = span.events
            # assert "completion_tokens" in span_metadata.attributes
            # assert "prompt_tokens" in span_metadata.attributes
            # assert "total_tokens" in span_metadata.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.invocation":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.name"] == "ReActAgent"
            assert span_attributes["entity.1.type"] == "agent.llamaindex"
            found_agent_span = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.tool.invocation":
            assert span_attributes["entity.1.type"] == "tool.llamaindex"
            assert span_attributes["entity.1.name"] == "place_order"
            found_tool_span = True

    assert found_inference_span, "Inference span not found"
    assert found_agent_span, "Agent span not found"
    assert found_tool_span, "Tool span not found"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# [{
#     "name": "openai.resources.chat.completions.Completions.create",
#     "context": {
#         "trace_id": "0x1d1e25b96aa3c89ebd7f9a58163069e8",
#         "span_id": "0x1aadf28593ef4f40",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6d62171ae9854915",
#     "start_time": "2025-07-14T03:09:24.838303Z",
#     "end_time": "2025-07-14T03:09:27.094057Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/llms/openai/base.py:476",
#         "workflow.name": "llama_index_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.llms.openai.base.OpenAI.chat",
#     "context": {
#         "trace_id": "0x1d1e25b96aa3c89ebd7f9a58163069e8",
#         "span_id": "0x6d62171ae9854915",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xdb87968eb29baf13",
#     "start_time": "2025-07-14T03:09:24.819159Z",
#     "end_time": "2025-07-14T03:09:27.094947Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/react/step.py:580",
#         "workflow.name": "llama_index_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1",
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T03:09:27.094830Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.\\n\\n## Tools\\n\\nYou have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.\\nThis may require breaking the task into subtasks and using different tools to complete each subtask.\\n\\nYou have access to the following tools:\\n> Tool Name: get_coffee_menu\\nTool Description: Provides a list of available coffee options with prices.\\nTool Args: {\\\"properties\\\": {}, \\\"type\\\": \\\"object\\\"}\\n\\n> Tool Name: place_order\\nTool Description: Takes a coffee order and provides the total cost.\\nTool Args: {\\\"properties\\\": {\\\"coffee_type\\\": {\\\"title\\\": \\\"Coffee Type\\\", \\\"type\\\": \\\"string\\\"}, \\\"quantity\\\": {\\\"title\\\": \\\"Quantity\\\", \\\"type\\\": \\\"integer\\\"}}, \\\"required\\\": [\\\"coffee_type\\\", \\\"quantity\\\"], \\\"type\\\": \\\"object\\\"}\\n\\n\\n\\n## Output Format\\n\\nPlease answer in the same language as the question and use the following format:\\n\\n```\\nThought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.\\nAction: tool name (one of get_coffee_menu, place_order) if using a tool.\\nAction Input: the input to the tool, in a JSON format representing the kwargs (e.g. {\\\"input\\\": \\\"hello world\\\", \\\"num_beams\\\": 5})\\n```\\n\\nPlease ALWAYS start with a Thought.\\n\\nNEVER surround your response with markdown code markers. You may use code markers within your response if you need to.\\n\\nPlease use a valid JSON format for the Action Input. Do NOT do this {'input': 'hello world', 'num_beams': 5}. If you include the \\\"Action:\\\" line, then you MUST include the \\\"Action Input:\\\" line too, even if the tool does not need kwargs, in that case you MUST use \\\"Action Input: {}\\\".\\n\\nIf this format is used, the tool will respond in the following format:\\n\\n```\\nObservation: tool response\\n```\\n\\nYou should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:\\n\\n```\\nThought: I can answer without using any more tools. I'll use the user's language to answer\\nAnswer: [your answer here (In the same language as the user's question)]\\n```\\n\\n```\\nThought: I cannot answer the question with the provided tools.\\nAnswer: [your answer here (In the same language as the user's question)]\\n```\\n\\n## Current Conversation\\n\\nBelow is the current conversation consisting of interleaving human and assistant messages.\\n\"}",
#                     "{\"user\": \"Please order 3 espresso coffees\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T03:09:27.094902Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "{\"assistant\": \"Thought: The user wants to order 3 espresso coffees. I need to use the 'place_order' tool to place this order.\\n\\nAction: place_order\\nAction Input: {\\\"coffee_type\\\": \\\"espresso\\\", \\\"quantity\\\": 3}\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-14T03:09:27.094931Z",
#             "attributes": {
#                 "temperature": 0.1,
#                 "completion_tokens": 50,
#                 "prompt_tokens": 575,
#                 "total_tokens": 625
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.tools.function_tool.FunctionTool.call",
#     "context": {
#         "trace_id": "0x1d1e25b96aa3c89ebd7f9a58163069e8",
#         "span_id": "0xbda02dee2aa62c83",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xdb87968eb29baf13",
#     "start_time": "2025-07-14T03:09:27.101919Z",
#     "end_time": "2025-07-14T03:09:27.102249Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/react/step.py:302",
#         "workflow.name": "llama_index_1",
#         "span.type": "agentic.tool.invocation",
#         "entity.1.type": "tool.llamaindex",
#         "entity.1.name": "place_order",
#         "entity.1.description": "Takes a coffee order and provides the total cost.",
#         "entity.2.name": "ReactAgent",
#         "entity.2.type": "agent.llamaindex",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T03:09:27.102222Z",
#             "attributes": {
#                 "input": [
#                     "{'espresso', 'coffee_type'}",
#                     "{'quantity', 3}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T03:09:27.102237Z",
#             "attributes": {
#                 "response": "Your order for 3 espresso(s) is confirmed. Total cost: $7.50"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.Completions.create",
#     "context": {
#         "trace_id": "0x1d1e25b96aa3c89ebd7f9a58163069e8",
#         "span_id": "0xb2b5e0bd9fcee9ab",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3862ec222f3341cd",
#     "start_time": "2025-07-14T03:09:27.104057Z",
#     "end_time": "2025-07-14T03:09:29.349061Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/llms/openai/base.py:476",
#         "workflow.name": "llama_index_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.llms.openai.base.OpenAI.chat",
#     "context": {
#         "trace_id": "0x1d1e25b96aa3c89ebd7f9a58163069e8",
#         "span_id": "0x3862ec222f3341cd",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xdb87968eb29baf13",
#     "start_time": "2025-07-14T03:09:27.103473Z",
#     "end_time": "2025-07-14T03:09:29.350108Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/react/step.py:580",
#         "workflow.name": "llama_index_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1",
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T03:09:29.349966Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.\\n\\n## Tools\\n\\nYou have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.\\nThis may require breaking the task into subtasks and using different tools to complete each subtask.\\n\\nYou have access to the following tools:\\n> Tool Name: get_coffee_menu\\nTool Description: Provides a list of available coffee options with prices.\\nTool Args: {\\\"properties\\\": {}, \\\"type\\\": \\\"object\\\"}\\n\\n> Tool Name: place_order\\nTool Description: Takes a coffee order and provides the total cost.\\nTool Args: {\\\"properties\\\": {\\\"coffee_type\\\": {\\\"title\\\": \\\"Coffee Type\\\", \\\"type\\\": \\\"string\\\"}, \\\"quantity\\\": {\\\"title\\\": \\\"Quantity\\\", \\\"type\\\": \\\"integer\\\"}}, \\\"required\\\": [\\\"coffee_type\\\", \\\"quantity\\\"], \\\"type\\\": \\\"object\\\"}\\n\\n\\n\\n## Output Format\\n\\nPlease answer in the same language as the question and use the following format:\\n\\n```\\nThought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.\\nAction: tool name (one of get_coffee_menu, place_order) if using a tool.\\nAction Input: the input to the tool, in a JSON format representing the kwargs (e.g. {\\\"input\\\": \\\"hello world\\\", \\\"num_beams\\\": 5})\\n```\\n\\nPlease ALWAYS start with a Thought.\\n\\nNEVER surround your response with markdown code markers. You may use code markers within your response if you need to.\\n\\nPlease use a valid JSON format for the Action Input. Do NOT do this {'input': 'hello world', 'num_beams': 5}. If you include the \\\"Action:\\\" line, then you MUST include the \\\"Action Input:\\\" line too, even if the tool does not need kwargs, in that case you MUST use \\\"Action Input: {}\\\".\\n\\nIf this format is used, the tool will respond in the following format:\\n\\n```\\nObservation: tool response\\n```\\n\\nYou should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:\\n\\n```\\nThought: I can answer without using any more tools. I'll use the user's language to answer\\nAnswer: [your answer here (In the same language as the user's question)]\\n```\\n\\n```\\nThought: I cannot answer the question with the provided tools.\\nAnswer: [your answer here (In the same language as the user's question)]\\n```\\n\\n## Current Conversation\\n\\nBelow is the current conversation consisting of interleaving human and assistant messages.\\n\"}",
#                     "{\"user\": \"Please order 3 espresso coffees\"}",
#                     "{\"assistant\": \"Thought: The user wants to order 3 espresso coffees. I need to use the 'place_order' tool to place this order.\\nAction: place_order\\nAction Input: {'coffee_type': 'espresso', 'quantity': 3}\"}",
#                     "{\"user\": \"Observation: Your order for 3 espresso(s) is confirmed. Total cost: $7.50\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T03:09:29.350030Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "{\"assistant\": \"Thought: I can answer without using any more tools. I'll use the user's language to answer\\nAnswer: Your order for 3 espresso coffees is confirmed. The total cost is $7.50.\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-14T03:09:29.350089Z",
#             "attributes": {
#                 "temperature": 0.1,
#                 "completion_tokens": 43,
#                 "prompt_tokens": 654,
#                 "total_tokens": 697
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.ReActAgent.chat",
#     "context": {
#         "trace_id": "0x1d1e25b96aa3c89ebd7f9a58163069e8",
#         "span_id": "0xdb87968eb29baf13",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xb3f2e3475e27b14e",
#     "start_time": "2025-07-14T03:09:24.816570Z",
#     "end_time": "2025-07-14T03:09:29.353637Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/tests/integration/test_llamaindex_agent_sample.py:59",
#         "workflow.name": "llama_index_1",
#         "span.type": "agentic.invocation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.name": "ReActAgent",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T03:09:29.353557Z",
#             "attributes": {
#                 "input": [
#                     "Please order 3 espresso coffees"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T03:09:29.353604Z",
#             "attributes": {
#                 "response": "Your order for 3 espresso coffees is confirmed. The total cost is $7.50."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x1d1e25b96aa3c89ebd7f9a58163069e8",
#         "span_id": "0xb3f2e3475e27b14e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-14T03:09:24.816519Z",
#     "end_time": "2025-07-14T03:09:29.353801Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/tests/integration/test_llamaindex_agent_sample.py:59",
#         "span.type": "workflow",
#         "entity.1.name": "llama_index_1",
#         "entity.1.type": "workflow.llamaindex",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# ]