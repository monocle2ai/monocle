
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
        workflow_name="llamaindex_agent_1",
        span_processors=[SimpleSpanProcessor(memory_exporter)]
)

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

def setup_agents():
    llm = OpenAI(model="gpt-4")

    flight_tool = FunctionTool.from_defaults(
        fn=book_flight,
        name="book_flight",
        description="Books a flight from one airport to another."
    )
    flight_agent = FunctionAgent(name="flight_booking_agent", tools=[flight_tool], llm=llm,
                            system_prompt="You are a flight booking agent who books flights as per the request. Once you complete the task, you handoff to the coordinator agent only.",
                            description="Flight booking agent")

    hotel_tool = FunctionTool.from_defaults(
        fn=book_hotel,
        name="book_hotel",
        description="Books a hotel stay."
    )
    hotel_agent = FunctionAgent(name="hotel_booking_agent", tools=[hotel_tool], llm=llm,
                            system_prompt="You are a hotel booking agent who books hotels as per the request. Once you complete the task, you handoff to the coordinator agent only.",
                            description="Hotel booking agent")

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
            assert span_attributes["entity.2.name"] == "gpt-4"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4"

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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0xa0261e2d36f1531d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc4e0715cd30445ec",
#     "start_time": "2025-07-12T19:14:25.492849Z",
#     "end_time": "2025-07-12T19:14:27.368728Z",
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
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:14:26.497740Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a coordinator agent who manages the flight and hotel booking agents. Separate hotel booking and flight booking tasks clearly from the input query. \\n                            Delegate only hotel booking to the hotel booking agent and only flight booking to the flight booking agent.\\n                            Once they complete their tasks, you collect their responses and provide consolidated response to the user.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:14:26.497740Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-12T19:14:27.366010Z",
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0xc4e0715cd30445ec",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:14:25.455919Z",
#     "end_time": "2025-07-12T19:14:27.369562Z",
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
# },
# {
#     "name": "llama_index.core.agent.workflow.multi_agent_workflow.AgentWorkflow._call_tool",
#     "context": {
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x2110abe36f79da64",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:14:33.472275Z",
#     "end_time": "2025-07-12T19:14:55.930356Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:447",
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
# },
# {
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.finalize",
#     "context": {
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x40ca4d9b4754e462",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:02.484539Z",
#     "end_time": "2025-07-12T19:15:02.488984Z",
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
#         "entity.1.description": "Travel booking coordinator agent",
#         "entity.1.type": "agent.llamaindex",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:02.488361Z",
#             "attributes": {
#                 "input": [
#                     "Agent flight_booking_agent is now handling the request due to the following reason: Book a flight from BOS to JFK.\nPlease continue with the current request."
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:02.488851Z",
#             "attributes": {
#                 "response": "Agent flight_booking_agent is now handling the request due to the following reason: Book a flight from BOS to JFK.\nPlease continue with the current request."
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x086f4caf7fc5ad38",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xbb6779a172894c29",
#     "start_time": "2025-07-12T19:15:02.503153Z",
#     "end_time": "2025-07-12T19:15:04.231581Z",
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
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:03.435830Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a flight booking agent who books flights as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: Book a flight from BOS to JFK.\\nPlease continue with the current request.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:03.435830Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-12T19:15:04.230045Z",
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0xbb6779a172894c29",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:02.498010Z",
#     "end_time": "2025-07-12T19:15:04.232132Z",
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
#     "name": "llama_index.core.agent.workflow.multi_agent_workflow.AgentWorkflow._call_tool",
#     "context": {
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0xc9555cba56e85d18",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:04.236079Z",
#     "end_time": "2025-07-12T19:15:09.718235Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:447",
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
#             "timestamp": "2025-07-12T19:15:09.717311Z",
#             "attributes": {
#                 "Inputs": [
#                     "{'BOS', 'from_airport'}",
#                     "{'JFK', 'to_airport'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:09.718054Z",
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
# },
# {
#     "name": "openai.resources.chat.completions.AsyncCompletions.create",
#     "context": {
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x2ff2cfd23044e376",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x021c9a78df1ae705",
#     "start_time": "2025-07-12T19:15:09.738026Z",
#     "end_time": "2025-07-12T19:15:11.228498Z",
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
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:10.405812Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a flight booking agent who books flights as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: Book a flight from BOS to JFK.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:10.405812Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-12T19:15:11.227508Z",
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x021c9a78df1ae705",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:09.733525Z",
#     "end_time": "2025-07-12T19:15:11.229033Z",
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
# },
# {
#     "name": "llama_index.core.agent.workflow.multi_agent_workflow.AgentWorkflow._call_tool",
#     "context": {
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0xddf090fd8e8e8a42",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:11.233315Z",
#     "end_time": "2025-07-12T19:15:22.206629Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:447",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.delegation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.from_agent": "flight_booking_agent",
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x844bb735a4fd7e65",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:22.215985Z",
#     "end_time": "2025-07-12T19:15:22.220066Z",
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
#         "entity.1.description": "Flight booking agent",
#         "entity.1.type": "agent.llamaindex",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:22.219502Z",
#             "attributes": {
#                 "input": [
#                     "Agent hotel_booking_agent is now handling the request due to the following reason: Book a hotel stay at McKittrick Hotel.\nPlease continue with the current request."
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:22.220040Z",
#             "attributes": {
#                 "response": "Agent hotel_booking_agent is now handling the request due to the following reason: Book a hotel stay at McKittrick Hotel.\nPlease continue with the current request."
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0xbb0d054c3365cc20",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x841d9d55e2cd0d5a",
#     "start_time": "2025-07-12T19:15:22.232836Z",
#     "end_time": "2025-07-12T19:15:23.686724Z",
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
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:23.093350Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a hotel booking agent who books hotels as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: Book a flight from BOS to JFK.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: Book a hotel stay at McKittrick Hotel.\\nPlease continue with the current request.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:23.093350Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-12T19:15:23.685462Z",
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x841d9d55e2cd0d5a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:22.227957Z",
#     "end_time": "2025-07-12T19:15:23.687101Z",
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
#     "name": "llama_index.core.agent.workflow.multi_agent_workflow.AgentWorkflow._call_tool",
#     "context": {
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0xdcb0f293307f3df8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:23.689802Z",
#     "end_time": "2025-07-12T19:15:23.691317Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:447",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.tool",
#         "entity.1.type": "tool.llamaindex",
#         "entity.1.name": "book_hotel",
#         "entity.1.description": "Books a hotel stay.",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:23.690987Z",
#             "attributes": {
#                 "Inputs": [
#                     "{'hotel_name', 'McKittrick Hotel'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:23.691300Z",
#             "attributes": {
#                 "response": "Successfully booked a stay at McKittrick Hotel."
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0xfca4258cf8b6e1c9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x868d927795b406bc",
#     "start_time": "2025-07-12T19:15:23.696385Z",
#     "end_time": "2025-07-12T19:15:25.490879Z",
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
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:24.934043Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a hotel booking agent who books hotels as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: Book a flight from BOS to JFK.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: Book a hotel stay at McKittrick Hotel.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a stay at McKittrick Hotel.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:24.934043Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-12T19:15:25.490309Z",
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x868d927795b406bc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:23.694390Z",
#     "end_time": "2025-07-12T19:15:25.491124Z",
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
#     "name": "llama_index.core.agent.workflow.multi_agent_workflow.AgentWorkflow._call_tool",
#     "context": {
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0xb7924d54de8ac529",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:25.493581Z",
#     "end_time": "2025-07-12T19:15:25.494505Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:447",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.delegation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.from_agent": "hotel_booking_agent",
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x0f9a1cabc3260a45",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:25.495953Z",
#     "end_time": "2025-07-12T19:15:25.497021Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:522",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.name": "hotel_booking_agent",
#         "entity.1.description": "Hotel booking agent",
#         "entity.1.type": "agent.llamaindex",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:25.496769Z",
#             "attributes": {
#                 "input": [
#                     "Agent coordinator is now handling the request due to the following reason: Completed booking tasks.\nPlease continue with the current request."
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:25.497008Z",
#             "attributes": {
#                 "response": "Agent coordinator is now handling the request due to the following reason: Completed booking tasks.\nPlease continue with the current request."
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x37a38e9e7d4b9417",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x93c4b3bf0591fbeb",
#     "start_time": "2025-07-12T19:15:25.501062Z",
#     "end_time": "2025-07-12T19:15:27.132931Z",
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
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:26.223975Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a coordinator agent who manages the flight and hotel booking agents. Separate hotel booking and flight booking tasks clearly from the input query. \\n                            Delegate only hotel booking to the hotel booking agent and only flight booking to the flight booking agent.\\n                            Once they complete their tasks, you collect their responses and provide consolidated response to the user.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: Book a flight from BOS to JFK.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: Book a hotel stay at McKittrick Hotel.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a stay at McKittrick Hotel.\"}",
#                     "{\"tool\": \"Agent coordinator is now handling the request due to the following reason: Completed booking tasks.\\nPlease continue with the current request.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:26.359208Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"I have successfully booked your flight from BOS to JFK and your stay at the McKittrick Hotel. Safe travels!\"}",
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-12T19:15:27.132207Z",
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x93c4b3bf0591fbeb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:25.499512Z",
#     "end_time": "2025-07-12T19:15:27.133331Z",
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0xdb263020b407e4eb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65d0e590c144c230",
#     "start_time": "2025-07-12T19:15:27.135107Z",
#     "end_time": "2025-07-12T19:15:27.136725Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:399",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.name": "coordinator",
#         "entity.1.description": "Travel booking coordinator agent",
#         "entity.1.type": "agent.llamaindex",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:27.136356Z",
#             "attributes": {}
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:27.136703Z",
#             "attributes": {
#                 "response": "I have successfully booked your flight from BOS to JFK and your stay at the McKittrick Hotel. Safe travels!"
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
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x65d0e590c144c230",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x681ccb3ecd5a7f13",
#     "start_time": "2025-07-12T19:14:25.388664Z",
#     "end_time": "2025-07-12T19:15:52.103310Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/tests/integration/test_llamaindex_multi_agent.py:75",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.request",
#         "entity.1.type": "agent.llamaindex",
#         "entity.count": 1,
#         "_active_agent_name": ""
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-12T19:15:39.862247Z",
#             "attributes": {
#                 "input": "book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-12T19:15:52.102484Z",
#             "attributes": {
#                 "response": "I have successfully booked your flight from BOS to JFK and your stay at the McKittrick Hotel. Safe travels!"
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
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xf0780895df20f97a5ab8bb7bcbf7d808",
#         "span_id": "0x681ccb3ecd5a7f13",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-12T19:14:25.388158Z",
#     "end_time": "2025-07-12T19:15:52.103710Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/tests/integration/test_llamaindex_multi_agent.py:75",
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