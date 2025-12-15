
import asyncio
import logging
import time

import pytest
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter


logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup():
    memory_exporter = InMemorySpanExporter()
    file_exporter = FileSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="llamaindex_agent_1",
            span_processors=[
                SimpleSpanProcessor(memory_exporter),
                BatchSpanProcessor(file_exporter),
            ]
        )
        yield memory_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

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
    logger.info(resp)

def test_multi_agent(setup):
    """Test multi-agent interaction with flight and hotel booking."""

    asyncio.run(run_agent())
    verify_spans(memory_exporter=setup)

def verify_spans(memory_exporter = None):
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
            
             # Check for delegation via from_agent attribute
            if "entity.1.from_agent" in span_attributes:
                from_agent = span_attributes["entity.1.from_agent"]
                agent_name = span_attributes["entity.1.name"]
                if agent_name == "flight_booking_agent" and from_agent == "coordinator":
                    found_book_flight_delegation = True
                elif agent_name == "hotel_booking_agent" and from_agent == "flight_booking_agent":
                    found_book_hotel_delegation = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.tool.invocation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert "entity.2.type" in span_attributes
            assert "entity.2.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.llamaindex"
            if span_attributes["entity.1.name"] == "book_flight" and span_attributes["entity.2.name"] == "flight_booking_agent":
                found_book_flight_tool = True
            elif span_attributes["entity.1.name"] == "book_hotel" and span_attributes["entity.2.name"] == "hotel_booking_agent":
                found_book_hotel_tool = True
            found_tool = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    assert found_flight_agent, "Flight assistant agent span not found"
    assert found_hotel_agent, "Hotel assistant agent span not found"
    assert found_supervisor_agent, "Supervisor agent span not found"
    assert found_book_flight_tool, "Book flight tool span not found"
    assert found_book_hotel_tool, "Book hotel tool span not found"
    assert found_book_flight_delegation, "Book flight delegation span not found (check from_agent attribute)"
    assert found_book_hotel_delegation, "Book hotel delegation span not found (check from_agent attribute)"



# [{
#     "name": "openai.resources.chat.completions.AsyncCompletions.create",
#     "context": {
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x398f1a20da2f3f32",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x88f13c3fbb368053",
#     "start_time": "2025-07-14T02:07:57.735677Z",
#     "end_time": "2025-07-14T02:07:59.662572Z",
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
#             "timestamp": "2025-07-14T02:07:58.708251Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a coordinator agent who manages the flight and hotel booking agents. Separate hotel booking and flight booking tasks clearly from the input query. \\n                            Delegate only hotel booking to the hotel booking agent and only flight booking to the flight booking agent.\\n                            Once they complete their tasks, you collect their responses and provide consolidated response to the user.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:07:58.708251Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-14T02:07:59.662255Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x88f13c3fbb368053",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:07:57.716919Z",
#     "end_time": "2025-07-14T02:07:59.662897Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0xc240afa80dc81067",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:07:59.664711Z",
#     "end_time": "2025-07-14T02:07:59.664945Z",
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
# }
# ,{
#     "name": "llama_index.core.agent.workflow.function_agent.FunctionAgent.finalize",
#     "context": {
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0xc1363b5375d34c93",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:07:59.665604Z",
#     "end_time": "2025-07-14T02:07:59.665990Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:522",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.name": "coordinator",
#         "entity.1.description": "Travel booking coordinator agent",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T02:07:59.665953Z",
#             "attributes": {
#                 "input": [
#                     "Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK.\nPlease continue with the current request."
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:07:59.665980Z",
#             "attributes": {
#                 "response": "Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK.\nPlease continue with the current request."
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
# {
#     "name": "openai.resources.chat.completions.AsyncCompletions.create",
#     "context": {
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x5fdf0516e20fba70",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x8defa99101a1da44",
#     "start_time": "2025-07-14T02:07:59.670765Z",
#     "end_time": "2025-07-14T02:08:01.693516Z",
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
#             "timestamp": "2025-07-14T02:08:00.971707Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a flight booking agent who books flights as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK.\\nPlease continue with the current request.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:00.971707Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-14T02:08:01.693214Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x8defa99101a1da44",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:07:59.667425Z",
#     "end_time": "2025-07-14T02:08:01.693817Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x54e80ad0ae022cf2",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:01.696293Z",
#     "end_time": "2025-07-14T02:08:01.696920Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:447",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.tool.invocation",
#         "entity.1.type": "tool.llamaindex",
#         "entity.1.name": "book_flight",
#         "entity.1.description": "Books a flight from one airport to another.",
#         "entity.2.name": "flight_booking_agent",
#         "entity.2.type": "agent.langgraph",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T02:08:01.696888Z",
#             "attributes": {
#                 "input": [
#                     "{'BOS', 'from_airport'}",
#                     "{'to_airport', 'JFK'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:01.696907Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x6cc484c6bd5bb4b0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x713277d415f9cd95",
#     "start_time": "2025-07-14T02:08:01.702022Z",
#     "end_time": "2025-07-14T02:08:03.361891Z",
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
#             "timestamp": "2025-07-14T02:08:02.519043Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a flight booking agent who books flights as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:02.519043Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-14T02:08:03.361524Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x713277d415f9cd95",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:01.699420Z",
#     "end_time": "2025-07-14T02:08:03.362366Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0xe48d8fdd14e1352c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:03.366574Z",
#     "end_time": "2025-07-14T02:08:03.366938Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0xae70f70298f04c90",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:03.368342Z",
#     "end_time": "2025-07-14T02:08:03.368977Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:522",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.name": "flight_booking_agent",
#         "entity.1.description": "Flight booking agent",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T02:08:03.368936Z",
#             "attributes": {
#                 "input": [
#                     "Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel.\nPlease continue with the current request."
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:03.368964Z",
#             "attributes": {
#                 "response": "Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel.\nPlease continue with the current request."
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
# {
#     "name": "openai.resources.chat.completions.AsyncCompletions.create",
#     "context": {
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x789bfe5827c890c1",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x8f99a1306e774342",
#     "start_time": "2025-07-14T02:08:03.373760Z",
#     "end_time": "2025-07-14T02:08:05.112802Z",
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
#             "timestamp": "2025-07-14T02:08:04.682406Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a hotel booking agent who books hotels as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel.\\nPlease continue with the current request.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:04.682406Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-14T02:08:05.112567Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x8f99a1306e774342",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:03.371349Z",
#     "end_time": "2025-07-14T02:08:05.113073Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x4506b56770219e44",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:05.115297Z",
#     "end_time": "2025-07-14T02:08:05.116014Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:447",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.tool.invocation",
#         "entity.1.type": "tool.llamaindex",
#         "entity.1.name": "book_hotel",
#         "entity.1.description": "Books a hotel stay.",
#         "entity.2.name": "hotel_booking_agent",
#         "entity.2.type": "agent.langgraph",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T02:08:05.115983Z",
#             "attributes": {
#                 "input": [
#                     "{'McKittrick Hotel', 'hotel_name'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:05.116001Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x41a5031e3138a3f9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x93038102a7be3876",
#     "start_time": "2025-07-14T02:08:05.120507Z",
#     "end_time": "2025-07-14T02:08:07.151367Z",
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
#             "timestamp": "2025-07-14T02:08:06.420972Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a hotel booking agent who books hotels as per the request. Once you complete the task, you handoff to the coordinator agent only.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a stay at McKittrick Hotel.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:06.420972Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-14T02:08:07.151026Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x93038102a7be3876",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:05.118605Z",
#     "end_time": "2025-07-14T02:08:07.151802Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x21563fddf5958bbf",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:07.154828Z",
#     "end_time": "2025-07-14T02:08:07.155189Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0xc6e3cef9c6e24d77",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:07.156648Z",
#     "end_time": "2025-07-14T02:08:07.158086Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:522",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.name": "hotel_booking_agent",
#         "entity.1.description": "Hotel booking agent",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T02:08:07.158040Z",
#             "attributes": {
#                 "input": [
#                     "Agent coordinator is now handling the request due to the following reason: Completed booking tasks for flight and hotel. Handing off to coordinator for further assistance..\nPlease continue with the current request."
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:07.158071Z",
#             "attributes": {
#                 "response": "Agent coordinator is now handling the request due to the following reason: Completed booking tasks for flight and hotel. Handing off to coordinator for further assistance..\nPlease continue with the current request."
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x090e446c7bfaa7dc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf0a975ae8fe627d6",
#     "start_time": "2025-07-14T02:08:07.165963Z",
#     "end_time": "2025-07-14T02:08:08.932875Z",
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
#             "timestamp": "2025-07-14T02:08:07.876668Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a coordinator agent who manages the flight and hotel booking agents. Separate hotel booking and flight booking tasks clearly from the input query. \\n                            Delegate only hotel booking to the hotel booking agent and only flight booking to the flight booking agent.\\n                            Once they complete their tasks, you collect their responses and provide consolidated response to the user.\"}",
#                     "{\"user\": \"book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel\"}",
#                     "{\"tool\": \"Agent flight_booking_agent is now handling the request due to the following reason: User requested to book a flight from BOS to JFK.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}",
#                     "{\"tool\": \"Agent hotel_booking_agent is now handling the request due to the following reason: User requested to book a hotel stay at McKittrick Hotel.\\nPlease continue with the current request.\"}",
#                     "{\"tool\": \"Successfully booked a stay at McKittrick Hotel.\"}",
#                     "{\"tool\": \"Agent coordinator is now handling the request due to the following reason: Completed booking tasks for flight and hotel. Handing off to coordinator for further assistance..\\nPlease continue with the current request.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:07.974635Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"I have successfully booked your flight from BOS to JFK and your stay at McKittrick Hotel. If you need any further assistance, feel free to ask.\"}",
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-14T02:08:08.932528Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0xf0a975ae8fe627d6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:07.161933Z",
#     "end_time": "2025-07-14T02:08:08.933214Z",
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0xf763a2d755b81b6b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x582cd8bfaf84d9bb",
#     "start_time": "2025-07-14T02:08:08.934868Z",
#     "end_time": "2025-07-14T02:08:08.935488Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/llama_index/core/agent/workflow/multi_agent_workflow.py:399",
#         "workflow.name": "llamaindex_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.type": "agent.llamaindex",
#         "entity.1.name": "coordinator",
#         "entity.1.description": "Travel booking coordinator agent",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T02:08:08.935442Z",
#             "attributes": {}
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:08.935473Z",
#             "attributes": {
#                 "response": "I have successfully booked your flight from BOS to JFK and your stay at McKittrick Hotel. If you need any further assistance, feel free to ask."
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x582cd8bfaf84d9bb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x8f24cb4ea2bc4709",
#     "start_time": "2025-07-14T02:07:57.652168Z",
#     "end_time": "2025-07-14T02:08:08.938174Z",
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
#             "timestamp": "2025-07-14T02:08:08.938092Z",
#             "attributes": {
#                 "input": "book a flight from BOS to JFK and a book hotel stay at McKittrick Hotel"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T02:08:08.938130Z",
#             "attributes": {
#                 "response": "I have successfully booked your flight from BOS to JFK and your stay at McKittrick Hotel. If you need any further assistance, feel free to ask."
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
#         "trace_id": "0x7cce9fed64277176ebeeb197363c22ee",
#         "span_id": "0x8f24cb4ea2bc4709",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-14T02:07:57.652120Z",
#     "end_time": "2025-07-14T02:08:08.938224Z",
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