
import asyncio
import time
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from llama_index.core.agent import ReActAgent
import logging
logger = logging.getLogger(__name__)

import time
import pytest

memory_exporter = InMemorySpanExporter()
span_processors=[SimpleSpanProcessor(memory_exporter)]

@pytest.fixture(scope="module")
def setup():
    memory_exporter.clear()
    setup_monocle_telemetry(
            workflow_name="llamaindex_agent_1", # monocle_exporters_list='file',
           span_processors=[SimpleSpanProcessor(memory_exporter)]
)

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

def setup_agents():
    llm = OpenAI(model="gpt-4o-mini")

    flight_tool = FunctionTool.from_defaults(
        fn=book_flight,
        name="book_flight",
        description="Books a flight from one airport to another."
    )
    flight_agent = FunctionAgent(name="flight_booking_agent", tools=[flight_tool], llm=llm,
                            system_prompt="You are a flight booking agent who books flights as per the request. Once you complete the task, you handoff to the coordinator agent only.",
                            description="Flight booking agent",
                            can_handoff_to=["coordinator"])

    hotel_tool = FunctionTool.from_defaults(
        fn=book_hotel,
        name="book_hotel",
        description="Books a hotel stay."
    )
    hotel_agent = FunctionAgent(name="hotel_booking_agent", tools=[hotel_tool], llm=llm,
                            system_prompt="You are a hotel booking agent who books hotels as per the request. Once you complete the task, you handoff to the coordinator agent only.",
                            description="Hotel booking agent",
                            can_handoff_to=["coordinator"])

    coordinator = FunctionAgent(name="coordinator", tools=[], llm=llm,
                            system_prompt=
                            """You are a coordinator agent who manages the flight and hotel booking agents. Separate hotel booking and flight booking tasks clearly from the input query. 
                            Delegate only hotel booking to the hotel booking agent and only flight booking to the flight booking agent.
                            Once they complete their tasks, you collect their responses and provide consolidated response to the user.""",
                            description="Travel booking coordinator agent",
                            can_handoff_to=["flight_booking_agent", "hotel_booking_agent"])

    agent_workflow = AgentWorkflow(
        agents=[coordinator, flight_agent, hotel_agent],
        root_agent=coordinator.name
    )
    return agent_workflow

async def run_agent():
    """Test multi-agent interaction with flight and hotel booking."""

    agent_workflow = setup_agents()
    resp = await agent_workflow.run(user_msg="book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel")
    print(resp)

@pytest.mark.integration()
def test_multi_agent(setup):
    """Test multi-agent interaction with flight and hotel booking."""

    asyncio.run(run_agent())
    verify_spans()

def verify_spans():
    time.sleep(2)
    found_inference = found_agent = found_tool = False
    found_flight_agent = found_hotel_agent = found_supervisor_agent = False
    found_book_hotel_tool = found_book_flight_tool = False
    found_book_flight_delegation = found_book_hotel_delegation = False
    spans = memory_exporter.get_finished_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4o-mini"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4o-mini"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
### TODO: Handle streaming responses from OpenAI LLM called by LlamaIndex agentic workflow
##            assert "completion_tokens" in span_metadata.attributes
##            assert "prompt_tokens" in span_metadata.attributes
##            assert "total_tokens" in span_metadata.attributes
            found_inference = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.invocation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.llamaindex"
            if span_attributes["entity.1.name"] == "flight_booking_agent":
                found_flight_agent = True
            elif span_attributes["entity.1.name"] == "hotel_booking_agent":
                found_hotel_agent = True
            elif span_attributes["entity.1.name"] == "coordinator":
                found_supervisor_agent = True
            found_agent = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.tool":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.llamaindex"
            if span_attributes["entity.1.name"] == "book_flight":
                found_book_flight_tool = True
            elif span_attributes["entity.1.name"] == "book_hotel":
                found_book_hotel_tool = True
            found_tool = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.delegation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.from_agent" in span_attributes
            assert "entity.1.to_agent" in span_attributes

            assert span_attributes["entity.1.type"] == "agent.llamaindex"
            if span_attributes["entity.1.to_agent"] == "flight_booking_agent":
                found_book_flight_delegation = True
            elif span_attributes["entity.1.to_agent"] == "hotel_booking_agent":
                found_book_hotel_delegation = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    assert found_book_flight_delegation, "Book flight delegation span not found"
    assert found_book_hotel_delegation, "Book hotel delegation span not found"
    assert found_flight_agent, "Flight assistant agent span not found"
    assert found_hotel_agent, "Hotel assistant agent span not found"
    assert found_supervisor_agent, "Supervisor agent span not found"
    assert found_book_flight_tool, "Book flight tool span not found"
    assert found_book_hotel_tool, "Book hotel tool span not found"


# [{
#     "name": "openai.resources.chat.completions.AsyncCompletions.create",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x7b8b9824bec67d59",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9a438bed8c10cf77",
#     "start_time": "2025-07-10T03:42:17.016843Z",
#     "end_time": "2025-07-10T03:42:19.292420Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/llms/openai/base.py:783",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:19.250602Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a coordinator agent who manages the flight and hotel booking agents. Separate hotel booking and flight booking tasks clearly from the input query. \\n                            Delegate only hotel booking to the hotel booking agent and only flight booking to the flight booking agent.\\n                            Once they complete their tasks, you collect their responses and provide consolidated response to the user.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:19.250602Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-10T03:42:19.292081Z",
#             "attributes": {}
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.take_step",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x9a438bed8c10cf77",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:16.998216Z",
#     "end_time": "2025-07-10T03:42:19.292738Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:382",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.tools.function_tool.FunctionTool.acall",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x7ea0ac2ad325684e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:19.294883Z",
#     "end_time": "2025-07-10T03:42:19.295147Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:280",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.delegation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.from_agent": "coordinator",
#         "entity.1.to_agent": "hotel_booking_agent",
#         "entity.count": 1
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.tools.function_tool.FunctionTool.acall",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x2c9de76864165da9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:19.296080Z",
#     "end_time": "2025-07-10T03:42:19.296270Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:280",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.delegation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.from_agent": "coordinator",
#         "entity.1.to_agent": "flight_booking_agent",
#         "entity.count": 1
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.finalize",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0xfa1f625a18fe4fdc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:19.297019Z",
#     "end_time": "2025-07-10T03:42:19.297432Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:522",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.name": "coordinator",
#         "entity.1.type": "agent.llamaindex",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:19.297393Z",
#             "attributes": {
#                 "input": [
#                     "Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel..\nPlease continue with the current request."
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:19.297421Z",
#             "attributes": {
#                 "response": "Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel..\nPlease continue with the current request."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.AsyncCompletions.create",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x5a9851c529526cbd",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xce3daae0484d05a5",
#     "start_time": "2025-07-10T03:42:19.301079Z",
#     "end_time": "2025-07-10T03:42:20.046911Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/llms/openai/base.py:783",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:19.862499Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a flight booking agent who books flights as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK..\\nPlease continue with the current request.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:19.862499Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-10T03:42:20.046433Z",
#             "attributes": {}
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.take_step",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0xce3daae0484d05a5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:19.299324Z",
#     "end_time": "2025-07-10T03:42:20.047516Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:382",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.tools.function_tool.FunctionTool.acall",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x09e2c0051be8a88f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:20.050850Z",
#     "end_time": "2025-07-10T03:42:20.051529Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:282",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.tool",
#         "entity.1.type": "tool.llamaindex",
#         "entity.1.name": "book_flight",
#         "entity.1.description": "Books a flight from one airport to another.",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:20.051488Z",
#             "attributes": {
#                 "Inputs": [
#                     "{'from_airport', 'BOS'}",
#                     "{'JFK', 'to_airport'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:20.051516Z",
#             "attributes": {
#                 "response": "Successfully booked a flight from BOS to JFK."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.AsyncCompletions.create",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x774ae1f8399d46a6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9e50d75f212ad521",
#     "start_time": "2025-07-10T03:42:20.057563Z",
#     "end_time": "2025-07-10T03:42:20.756206Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/llms/openai/base.py:783",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:20.476194Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a flight booking agent who books flights as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:20.476194Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-10T03:42:20.755865Z",
#             "attributes": {}
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.take_step",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x9e50d75f212ad521",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:20.054832Z",
#     "end_time": "2025-07-10T03:42:20.756532Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:382",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.tools.function_tool.FunctionTool.acall",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x71cd9966b76e5f26",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:20.759297Z",
#     "end_time": "2025-07-10T03:42:20.759619Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:280",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.delegation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.from_agent": "flight_booking_agent",
#         "entity.1.to_agent": "coordinator",
#         "entity.count": 1
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.finalize",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0xbd1aa982cf20ad9d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:20.761003Z",
#     "end_time": "2025-07-10T03:42:20.761635Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:522",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.name": "flight_booking_agent",
#         "entity.1.type": "agent.llamaindex",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:20.761595Z",
#             "attributes": {
#                 "input": [
#                     "Agent coordinator is now handling the request due to the following reason: Flight from BOS to JFK has been successfully booked, handing off to the coordinator..\nPlease continue with the current request."
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:20.761623Z",
#             "attributes": {
#                 "response": "Agent coordinator is now handling the request due to the following reason: Flight from BOS to JFK has been successfully booked, handing off to the coordinator..\nPlease continue with the current request."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.AsyncCompletions.create",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x4d9f057051f26bba",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x13b06f4d0c2970df",
#     "start_time": "2025-07-10T03:42:20.766444Z",
#     "end_time": "2025-07-10T03:42:21.736665Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/llms/openai/base.py:783",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:21.500345Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a coordinator agent who manages the flight and hotel booking agents. Separate hotel booking and flight booking tasks clearly from the input query. \\n                            Delegate only hotel booking to the hotel booking agent and only flight booking to the flight booking agent.\\n                            Once they complete their tasks, you collect their responses and provide consolidated response to the user.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}",
#                     "{\"tool\": \"Agent coordinator is now handling the request due to the following reason: Flight from BOS to JFK has been successfully booked, handing off to the coordinator..\\nPlease continue with the current request.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:21.500345Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-10T03:42:21.736472Z",
#             "attributes": {}
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.take_step",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x13b06f4d0c2970df",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:20.764317Z",
#     "end_time": "2025-07-10T03:42:21.736832Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:382",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.tools.function_tool.FunctionTool.acall",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x25b3ef5b99a96d64",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:21.738509Z",
#     "end_time": "2025-07-10T03:42:21.738699Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:280",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.delegation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.from_agent": "coordinator",
#         "entity.1.to_agent": "coordinator",
#         "entity.count": 1
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.finalize",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x8e5b71ec74a0a675",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:21.739325Z",
#     "end_time": "2025-07-10T03:42:21.739681Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:522",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.name": "coordinator",
#         "entity.1.type": "agent.llamaindex",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:21.739649Z",
#             "attributes": {
#                 "input": [
#                     "Agent coordinator cannot hand off to coordinator. Please select a valid agent to hand off to."
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:21.739671Z",
#             "attributes": {
#                 "response": "Agent coordinator cannot hand off to coordinator. Please select a valid agent to hand off to."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.AsyncCompletions.create",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0xc7c622f803084aa3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x5381694df36fd4a4",
#     "start_time": "2025-07-10T03:42:21.742668Z",
#     "end_time": "2025-07-10T03:42:22.660935Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/llms/openai/base.py:783",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:22.334229Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a coordinator agent who manages the flight and hotel booking agents. Separate hotel booking and flight booking tasks clearly from the input query. \\n                            Delegate only hotel booking to the hotel booking agent and only flight booking to the flight booking agent.\\n                            Once they complete their tasks, you collect their responses and provide consolidated response to the user.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}",
#                     "{\"tool\": \"Agent coordinator is now handling the request due to the following reason: Flight from BOS to JFK has been successfully booked, handing off to the coordinator..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Agent coordinator cannot hand off to coordinator. Please select a valid agent to hand off to.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:22.334229Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-10T03:42:22.660621Z",
#             "attributes": {}
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.take_step",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x5381694df36fd4a4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:21.741521Z",
#     "end_time": "2025-07-10T03:42:22.661240Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:382",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.tools.function_tool.FunctionTool.acall",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x399244d286822b3f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:22.663657Z",
#     "end_time": "2025-07-10T03:42:22.663993Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:280",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.delegation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.from_agent": "coordinator",
#         "entity.1.to_agent": "hotel_booking_agent",
#         "entity.count": 1
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.finalize",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x255fec60d0220470",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:22.665284Z",
#     "end_time": "2025-07-10T03:42:22.665776Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:522",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.name": "coordinator",
#         "entity.1.type": "agent.llamaindex",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:22.665739Z",
#             "attributes": {
#                 "input": [
#                     "Agent hotel_booking_agent is now handling the request due to the following reason: Hotel stay at McKittrick Hotel has been successfully booked, handing off to the hotel booking agent..\nPlease continue with the current request."
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:22.665765Z",
#             "attributes": {
#                 "response": "Agent hotel_booking_agent is now handling the request due to the following reason: Hotel stay at McKittrick Hotel has been successfully booked, handing off to the hotel booking agent..\nPlease continue with the current request."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.AsyncCompletions.create",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x469adccbad56e596",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9a4cdc07a205d067",
#     "start_time": "2025-07-10T03:42:22.669850Z",
#     "end_time": "2025-07-10T03:42:23.675884Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/llms/openai/base.py:783",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:23.241538Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a hotel booking agent who books hotels as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}",
#                     "{\"tool\": \"Agent coordinator is now handling the request due to the following reason: Flight from BOS to JFK has been successfully booked, handing off to the coordinator..\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Agent coordinator cannot hand off to coordinator. Please select a valid agent to hand off to.\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: Hotel stay at McKittrick Hotel has been successfully booked, handing off to the hotel booking agent..\\nPlease continue with the current request.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:23.243349Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"I have successfully booked a flight from BOS to JFK. However, I need to hand off the hotel booking for the McKittrick Hotel to the hotel booking agent. Please wait a moment while I do that.\"}",
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-10T03:42:23.675469Z",
#             "attributes": {}
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.take_step",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x9a4cdc07a205d067",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:22.667875Z",
#     "end_time": "2025-07-10T03:42:23.676218Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:382",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.finalize",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0x21bc2d77aaacdfe5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd19b4d682cee8bc5",
#     "start_time": "2025-07-10T03:42:23.677795Z",
#     "end_time": "2025-07-10T03:42:23.678388Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:399",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.name": "hotel_booking_agent",
#         "entity.1.type": "agent.llamaindex",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T03:42:23.678343Z",
#             "attributes": {}
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T03:42:23.678374Z",
#             "attributes": {
#                 "response": "I have successfully booked a flight from BOS to JFK. However, I need to hand off the hotel booking for the McKittrick Hotel to the hotel booking agent. Please wait a moment while I do that."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "llama_index.core.agent.workflow.multi_agent_workflow.AgentWorkflow.run",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0xd19b4d682cee8bc5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd927af865b6685f7",
#     "start_time": "2025-07-10T03:42:16.931616Z",
#     "end_time": "2025-07-10T03:42:23.679868Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/tests/integration/test_llamaindex_multi_agent.py:77",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "generic",
#         "_active_agent_name": ""
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x59dbb00c47c537978bee083bfad1a753",
#         "span_id": "0xd927af865b6685f7",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-10T03:42:16.931567Z",
#     "end_time": "2025-07-10T03:42:23.679901Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/tests/integration/test_llamaindex_multi_agent.py:77",
#         "span.type": "workflow",
#         "entity.1.name": "llamaindex_agent_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ]