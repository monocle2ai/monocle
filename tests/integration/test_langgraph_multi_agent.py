

import asyncio
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
import pytest
import os
import logging
logger = logging.getLogger(__name__)

import time
memory_exporter = InMemorySpanExporter()
span_processors=[SimpleSpanProcessor(memory_exporter)]

@pytest.fixture(scope="module")
def setup():
    memory_exporter.clear()
    setup_monocle_telemetry(
            workflow_name="langchain_agent_1", monocle_exporters_list='file'
#            span_processors=[SimpleSpanProcessor(memory_exporter)]
)

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

def setup_agents():

    flight_assistant = create_react_agent(
        model="openai:gpt-4o",
        tools=[book_flight],
        prompt="You are a flight booking assistant",
        name="flight_assistant"
    )

    hotel_assistant = create_react_agent(
        model="openai:gpt-4o",
        tools=[book_hotel],
        prompt="You are a hotel booking assistant",
        name="hotel_assistant"
    )

    supervisor = create_supervisor(
        agents=[flight_assistant, hotel_assistant],
        model=ChatOpenAI(model="gpt-4o"),
        prompt=(
            "You manage a hotel booking assistant and a"
            "flight booking assistant. Assign work to them."
        )
    ).compile()

    return supervisor

@pytest.mark.integration()
def test_multi_agent(setup):
    """Test multi-agent interaction with flight and hotel booking."""

    supervisor = setup_agents()
    chunk = supervisor.invoke(
        input ={
            "messages": [
                {
                    "role": "user",
                    "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
                }
            ]
        }
    )
    print(chunk)
    print("\n")
    verify_spans()

@pytest.mark.integration()
def test_async_multi_agent(setup):
    """Test multi-agent interaction with flight and hotel booking."""

    supervisor = setup_agents()
    chunk = asyncio.run(supervisor.ainvoke(
        input ={
            "messages": [
                {
                    "role": "user",
                    "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
                }
            ]
        }
    ))
    print(chunk)
    print("\n")
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
            assert span_attributes["entity.2.name"] == "gpt-4o"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4o"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
            found_inference = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.invocation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.langgraph"
            if span_attributes["entity.1.name"] == "flight_assistant":
                found_flight_agent = True
            elif span_attributes["entity.1.name"] == "hotel_assistant":
                found_hotel_agent = True
            elif span_attributes["entity.1.name"] == "supervisor":
                found_supervisor_agent = True
            found_agent = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.tool":
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.langgraph"
            if span_attributes["entity.1.name"] == "book_flight":
                found_book_flight_tool = True
            elif span_attributes["entity.1.name"] == "book_hotel":
                found_book_hotel_tool = True
            found_tool = True

        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.delegation":
            assert "entity.1.type" in span_attributes
            assert "entity.1.from_agent" in span_attributes
            assert "entity.1.to_agent" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.langgraph"
            if span_attributes["entity.1.to_agent"] == "flight_assistant":
                found_book_flight_delegation = True
            elif span_attributes["entity.1.to_agent"] == "hotel_assistant":
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
#     "name": "openai.resources.chat.completions.Completions.create",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0xb15184254b70c65e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x69d1bc071fc4572d",
#     "start_time": "2025-07-09T02:19:31.030572Z",
#     "end_time": "2025-07-09T02:19:32.312378Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.chat_models.base.BaseChatModel.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x69d1bc071fc4572d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6948281b1a18c908",
#     "start_time": "2025-07-09T02:19:31.028396Z",
#     "end_time": "2025-07-09T02:19:32.318060Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/runnables/base.py:5431",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o",
#         "entity.2.type": "model.llm.gpt-4o",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:32.317881Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You manage a hotel booking assistant and aflight booking assistant. Assign work to them.\"}",
#                     "{\"human\": \"book a flight from BOS to JFK and a stay at McKittrick Hotel\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:32.317943Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-09T02:19:32.318026Z",
#             "attributes": {
#                 "completion_tokens": 14,
#                 "prompt_tokens": 100,
#                 "total_tokens": 114
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.schema.runnable.RunnableSequence.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x6948281b1a18c908",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x30460fa5fc45d284",
#     "start_time": "2025-07-09T02:19:31.025699Z",
#     "end_time": "2025-07-09T02:19:32.318230Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/prebuilt/chat_agent_executor.py:507",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain_core.tools.base.BaseTool.run",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0xfdf47cbbccafb47a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x30460fa5fc45d284",
#     "start_time": "2025-07-09T02:19:32.323817Z",
#     "end_time": "2025-07-09T02:19:32.325132Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/tools/base.py:599",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "agentic.delegation",
#         "entity.1.type": "agent.langgraph",
#         "entity.1.from_agent": "supervisor",
#         "entity.1.to_agent": "transfer_to_flight_assistant",
#         "entity.count": 1
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langgraph.graph.state.CompiledStateGraph.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x30460fa5fc45d284",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0c7bcda7610dd38b",
#     "start_time": "2025-07-09T02:19:31.025208Z",
#     "end_time": "2025-07-09T02:19:32.329375Z",
#     "status": {
#         "status_code": "ERROR",
#         "description": "ParentCommand: Command(graph='supervisor:03b8f733-a84e-bff5-a4d9-3269abab97c5', update={'messages': [HumanMessage(content='book a flight from BOS to JFK and a stay at McKittrick Hotel', additional_kwargs={}, response_metadata={}, id='dba4c19e-ccef-4b98-9e4e-f0ac699050ef'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'function': {'arguments': '{}', 'name': 'transfer_to_flight_assistant'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 100, 'total_tokens': 114, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a288987b44', 'id': 'chatcmpl-BrEh9VKqTNR9zkeavIs8VA0kKVNdp', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, name='supervisor', id='run--73f157f9-bcfd-4e1b-9cdd-e9baee185dd7-0', tool_calls=[{'name': 'transfer_to_flight_assistant', 'args': {}, 'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'type': 'tool_call'}], usage_metadata={'input_tokens': 100, 'output_tokens': 14, 'total_tokens': 114, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='Successfully transferred to flight_assistant', name='transfer_to_flight_assistant', tool_call_id='call_AatZnRVOBzIEfrIyRoqYgYaX')], 'is_last_step': False, 'remaining_steps': 24}, goto='flight_assistant')"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/utils/runnable.py:623",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.type": "agent.langgraph",
#         "entity.1.name": "supervisor",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:32.325597Z",
#             "attributes": {}
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:32.325606Z",
#             "attributes": {}
#         },
#         {
#             "name": "exception",
#             "timestamp": "2025-07-09T02:19:32.329226Z",
#             "attributes": {
#                 "exception.type": "langgraph.errors.ParentCommand",
#                 "exception.message": "Command(graph='supervisor:03b8f733-a84e-bff5-a4d9-3269abab97c5', update={'messages': [HumanMessage(content='book a flight from BOS to JFK and a stay at McKittrick Hotel', additional_kwargs={}, response_metadata={}, id='dba4c19e-ccef-4b98-9e4e-f0ac699050ef'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'function': {'arguments': '{}', 'name': 'transfer_to_flight_assistant'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 100, 'total_tokens': 114, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a288987b44', 'id': 'chatcmpl-BrEh9VKqTNR9zkeavIs8VA0kKVNdp', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, name='supervisor', id='run--73f157f9-bcfd-4e1b-9cdd-e9baee185dd7-0', tool_calls=[{'name': 'transfer_to_flight_assistant', 'args': {}, 'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'type': 'tool_call'}], usage_metadata={'input_tokens': 100, 'output_tokens': 14, 'total_tokens': 114, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='Successfully transferred to flight_assistant', name='transfer_to_flight_assistant', tool_call_id='call_AatZnRVOBzIEfrIyRoqYgYaX')], 'is_last_step': False, 'remaining_steps': 24}, goto='flight_assistant')",
#                 "exception.stacktrace": "Traceback (most recent call last):\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/opentelemetry/trace/__init__.py\", line 589, in use_span\n    yield span\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/opentelemetry/sdk/trace/__init__.py\", line 1105, in start_as_current_span\n    yield span\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/src/monocle_apptrace/instrumentation/common/wrapper.py\", line 78, in monocle_wrapper_span_processor\n    return_value = wrapped(*args, **kwargs)\n                   ^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/pregel/__init__.py\", line 2843, in invoke\n    for chunk in self.stream(\n                 ^^^^^^^^^^^^\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/pregel/__init__.py\", line 2533, in stream\n    for _ in runner.tick(\n             ^^^^^^^^^^^^\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/pregel/write.py\", line 86, in _write\n    self.do_write(\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/pregel/write.py\", line 128, in do_write\n    write(_assemble_writes(writes))\n          ^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/pregel/write.py\", line 183, in _assemble_writes\n    if ww := w.mapper(w.value):\n             ^^^^^^^^^^^^^^^^^\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/graph/state.py\", line 1238, in _control_branch\n    raise ParentCommand(command)\nlanggraph.errors.ParentCommand: Command(graph='supervisor:03b8f733-a84e-bff5-a4d9-3269abab97c5', update={'messages': [HumanMessage(content='book a flight from BOS to JFK and a stay at McKittrick Hotel', additional_kwargs={}, response_metadata={}, id='dba4c19e-ccef-4b98-9e4e-f0ac699050ef'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'function': {'arguments': '{}', 'name': 'transfer_to_flight_assistant'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 100, 'total_tokens': 114, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a288987b44', 'id': 'chatcmpl-BrEh9VKqTNR9zkeavIs8VA0kKVNdp', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, name='supervisor', id='run--73f157f9-bcfd-4e1b-9cdd-e9baee185dd7-0', tool_calls=[{'name': 'transfer_to_flight_assistant', 'args': {}, 'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'type': 'tool_call'}], usage_metadata={'input_tokens': 100, 'output_tokens': 14, 'total_tokens': 114, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='Successfully transferred to flight_assistant', name='transfer_to_flight_assistant', tool_call_id='call_AatZnRVOBzIEfrIyRoqYgYaX')], 'is_last_step': False, 'remaining_steps': 24}, goto='flight_assistant')\n",
#                 "exception.escaped": "False"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.Completions.create",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x4c92b1362b98df38",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xee5a4de5ad97340a",
#     "start_time": "2025-07-09T02:19:32.333275Z",
#     "end_time": "2025-07-09T02:19:33.302890Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.chat_models.base.BaseChatModel.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0xee5a4de5ad97340a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9f3748b24d098776",
#     "start_time": "2025-07-09T02:19:32.332089Z",
#     "end_time": "2025-07-09T02:19:33.303825Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/runnables/base.py:5431",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o",
#         "entity.2.type": "model.llm.gpt-4o",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:33.303733Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a flight booking assistant\"}",
#                     "{\"human\": \"book a flight from BOS to JFK and a stay at McKittrick Hotel\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully transferred to flight_assistant\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:33.303780Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-09T02:19:33.303809Z",
#             "attributes": {
#                 "completion_tokens": 25,
#                 "prompt_tokens": 108,
#                 "total_tokens": 133
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.schema.runnable.RunnableSequence.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x9f3748b24d098776",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x16d53eaae81587e0",
#     "start_time": "2025-07-09T02:19:32.331495Z",
#     "end_time": "2025-07-09T02:19:33.303913Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/prebuilt/chat_agent_executor.py:507",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain_core.tools.base.BaseTool.run",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x6fa051b165091c48",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x16d53eaae81587e0",
#     "start_time": "2025-07-09T02:19:33.306145Z",
#     "end_time": "2025-07-09T02:19:33.307175Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/tools/base.py:599",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "agentic.tool",
#         "entity.1.type": "tool.langgraph",
#         "entity.1.name": "book_flight",
#         "entity.1.description": "Book a flight",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:33.307137Z",
#             "attributes": {
#                 "Inputs": [
#                     "BOS",
#                     "JFK"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:33.307161Z",
#             "attributes": {
#                 "response": "Successfully booked a flight from BOS to JFK.",
#                 "status": "success"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.Completions.create",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x00ce6e122ce40ef0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6b3501e536b07eb5",
#     "start_time": "2025-07-09T02:19:33.311924Z",
#     "end_time": "2025-07-09T02:19:35.480669Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.chat_models.base.BaseChatModel.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x6b3501e536b07eb5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x4e4d8ff68ab600f8",
#     "start_time": "2025-07-09T02:19:33.310519Z",
#     "end_time": "2025-07-09T02:19:35.481461Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/runnables/base.py:5431",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o",
#         "entity.2.type": "model.llm.gpt-4o",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:35.481369Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a flight booking assistant\"}",
#                     "{\"human\": \"book a flight from BOS to JFK and a stay at McKittrick Hotel\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully transferred to flight_assistant\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully booked a flight from BOS to JFK.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:35.481420Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "{\"ai\": \"I have successfully booked a flight for you from Boston Logan International Airport (BOS) to John F. Kennedy International Airport (JFK). Unfortunately, I cannot book a stay at McKittrick Hotel. You may want to contact the hotel directly or use a travel website for accommodation bookings. If you need further assistance, feel free to ask!\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-09T02:19:35.481446Z",
#             "attributes": {
#                 "completion_tokens": 71,
#                 "prompt_tokens": 155,
#                 "total_tokens": 226
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.schema.runnable.RunnableSequence.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x4e4d8ff68ab600f8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x16d53eaae81587e0",
#     "start_time": "2025-07-09T02:19:33.308912Z",
#     "end_time": "2025-07-09T02:19:35.481545Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/prebuilt/chat_agent_executor.py:507",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langgraph.graph.state.CompiledStateGraph.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x16d53eaae81587e0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0c7bcda7610dd38b",
#     "start_time": "2025-07-09T02:19:32.330512Z",
#     "end_time": "2025-07-09T02:19:35.482429Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph_supervisor/supervisor.py:93",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.type": "agent.langgraph",
#         "entity.1.name": "flight_assistant",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:35.482401Z",
#             "attributes": {
#                 "query": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:35.482417Z",
#             "attributes": {
#                 "response": "I have successfully booked a flight for you from Boston Logan International Airport (BOS) to John F. Kennedy International Airport (JFK). Unfortunately, I cannot book a stay at McKittrick Hotel. You may want to contact the hotel directly or use a travel website for accommodation bookings. If you need further assistance, feel free to ask!"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.Completions.create",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x914b228458db794e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x14cdf7239c51fa27",
#     "start_time": "2025-07-09T02:19:35.488765Z",
#     "end_time": "2025-07-09T02:19:36.298362Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.chat_models.base.BaseChatModel.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x14cdf7239c51fa27",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x5bfc11eed4c9932c",
#     "start_time": "2025-07-09T02:19:35.487523Z",
#     "end_time": "2025-07-09T02:19:36.299111Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/runnables/base.py:5431",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o",
#         "entity.2.type": "model.llm.gpt-4o",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:36.299029Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You manage a hotel booking assistant and aflight booking assistant. Assign work to them.\"}",
#                     "{\"human\": \"book a flight from BOS to JFK and a stay at McKittrick Hotel\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully transferred to flight_assistant\"}",
#                     "{\"ai\": \"I have successfully booked a flight for you from Boston Logan International Airport (BOS) to John F. Kennedy International Airport (JFK). Unfortunately, I cannot book a stay at McKittrick Hotel. You may want to contact the hotel directly or use a travel website for accommodation bookings. If you need further assistance, feel free to ask!\"}",
#                     "{\"ai\": \"Transferring back to supervisor\"}",
#                     "{\"tool\": \"Successfully transferred back to supervisor\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:36.299071Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-09T02:19:36.299095Z",
#             "attributes": {
#                 "completion_tokens": 14,
#                 "prompt_tokens": 260,
#                 "total_tokens": 274
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.schema.runnable.RunnableSequence.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x5bfc11eed4c9932c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd38f62d56b3f0a98",
#     "start_time": "2025-07-09T02:19:35.486576Z",
#     "end_time": "2025-07-09T02:19:36.299189Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/prebuilt/chat_agent_executor.py:507",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain_core.tools.base.BaseTool.run",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x537c17b48a4b2aa7",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd38f62d56b3f0a98",
#     "start_time": "2025-07-09T02:19:36.300947Z",
#     "end_time": "2025-07-09T02:19:36.302329Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/tools/base.py:599",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "agentic.delegation",
#         "entity.1.type": "agent.langgraph",
#         "entity.1.from_agent": "supervisor",
#         "entity.1.to_agent": "transfer_to_hotel_assistant",
#         "entity.count": 1
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langgraph.graph.state.CompiledStateGraph.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0xd38f62d56b3f0a98",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0c7bcda7610dd38b",
#     "start_time": "2025-07-09T02:19:35.484503Z",
#     "end_time": "2025-07-09T02:19:36.305606Z",
#     "status": {
#         "status_code": "ERROR",
#         "description": "ParentCommand: Command(graph='supervisor:ab56bb00-b70f-2b04-c124-4741d4eeb9af', update={'messages': [HumanMessage(content='book a flight from BOS to JFK and a stay at McKittrick Hotel', additional_kwargs={}, response_metadata={}, id='dba4c19e-ccef-4b98-9e4e-f0ac699050ef'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'function': {'arguments': '{}', 'name': 'transfer_to_flight_assistant'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 100, 'total_tokens': 114, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a288987b44', 'id': 'chatcmpl-BrEh9VKqTNR9zkeavIs8VA0kKVNdp', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, name='supervisor', id='run--73f157f9-bcfd-4e1b-9cdd-e9baee185dd7-0', tool_calls=[{'name': 'transfer_to_flight_assistant', 'args': {}, 'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'type': 'tool_call'}], usage_metadata={'input_tokens': 100, 'output_tokens': 14, 'total_tokens': 114, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='Successfully transferred to flight_assistant', name='transfer_to_flight_assistant', id='a16d817d-3a32-42c0-a67d-fedd8b353490', tool_call_id='call_AatZnRVOBzIEfrIyRoqYgYaX'), AIMessage(content='I have successfully booked a flight for you from Boston Logan International Airport (BOS) to John F. Kennedy International Airport (JFK). Unfortunately, I cannot book a stay at McKittrick Hotel. You may want to contact the hotel directly or use a travel website for accommodation bookings. If you need further assistance, feel free to ask!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 71, 'prompt_tokens': 155, 'total_tokens': 226, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a288987b44', 'id': 'chatcmpl-BrEhBETSr4kcCM1trW7ahFHhImU44', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, name='flight_assistant', id='run--7ddb5d7f-2f9e-4446-ae88-6bcf5f56d18d-0', usage_metadata={'input_tokens': 155, 'output_tokens': 71, 'total_tokens': 226, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), AIMessage(content='Transferring back to supervisor', additional_kwargs={}, response_metadata={'__is_handoff_back': True}, name='flight_assistant', id='b82588d1-cd69-4b9e-8403-bea80f909423', tool_calls=[{'name': 'transfer_back_to_supervisor', 'args': {}, 'id': '9bada7d7-3e76-4c91-a897-f770e31af9d7', 'type': 'tool_call'}]), ToolMessage(content='Successfully transferred back to supervisor', name='transfer_back_to_supervisor', id='81986903-1e5f-44c9-a024-7ed7b7289960', tool_call_id='9bada7d7-3e76-4c91-a897-f770e31af9d7'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_qogNT5MLJccSbrlLW1VnLNOM', 'function': {'arguments': '{}', 'name': 'transfer_to_hotel_assistant'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 260, 'total_tokens': 274, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_07871e2ad8', 'id': 'chatcmpl-BrEhD7HCUqv5Halb397x6GLHMu6x2', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, name='supervisor', id='run--abe9f693-71a9-4d8a-9b2b-914adddb8b50-0', tool_calls=[{'name': 'transfer_to_hotel_assistant', 'args': {}, 'id': 'call_qogNT5MLJccSbrlLW1VnLNOM', 'type': 'tool_call'}], usage_metadata={'input_tokens': 260, 'output_tokens': 14, 'total_tokens': 274, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='Successfully transferred to hotel_assistant', name='transfer_to_hotel_assistant', tool_call_id='call_qogNT5MLJccSbrlLW1VnLNOM')], 'is_last_step': False, 'remaining_steps': 24}, goto='hotel_assistant')"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/utils/runnable.py:623",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.type": "agent.langgraph",
#         "entity.1.name": "supervisor",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:36.303022Z",
#             "attributes": {}
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:36.303032Z",
#             "attributes": {}
#         },
#         {
#             "name": "exception",
#             "timestamp": "2025-07-09T02:19:36.305093Z",
#             "attributes": {
#                 "exception.type": "langgraph.errors.ParentCommand",
#                 "exception.message": "Command(graph='supervisor:ab56bb00-b70f-2b04-c124-4741d4eeb9af', update={'messages': [HumanMessage(content='book a flight from BOS to JFK and a stay at McKittrick Hotel', additional_kwargs={}, response_metadata={}, id='dba4c19e-ccef-4b98-9e4e-f0ac699050ef'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'function': {'arguments': '{}', 'name': 'transfer_to_flight_assistant'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 100, 'total_tokens': 114, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a288987b44', 'id': 'chatcmpl-BrEh9VKqTNR9zkeavIs8VA0kKVNdp', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, name='supervisor', id='run--73f157f9-bcfd-4e1b-9cdd-e9baee185dd7-0', tool_calls=[{'name': 'transfer_to_flight_assistant', 'args': {}, 'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'type': 'tool_call'}], usage_metadata={'input_tokens': 100, 'output_tokens': 14, 'total_tokens': 114, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='Successfully transferred to flight_assistant', name='transfer_to_flight_assistant', id='a16d817d-3a32-42c0-a67d-fedd8b353490', tool_call_id='call_AatZnRVOBzIEfrIyRoqYgYaX'), AIMessage(content='I have successfully booked a flight for you from Boston Logan International Airport (BOS) to John F. Kennedy International Airport (JFK). Unfortunately, I cannot book a stay at McKittrick Hotel. You may want to contact the hotel directly or use a travel website for accommodation bookings. If you need further assistance, feel free to ask!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 71, 'prompt_tokens': 155, 'total_tokens': 226, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a288987b44', 'id': 'chatcmpl-BrEhBETSr4kcCM1trW7ahFHhImU44', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, name='flight_assistant', id='run--7ddb5d7f-2f9e-4446-ae88-6bcf5f56d18d-0', usage_metadata={'input_tokens': 155, 'output_tokens': 71, 'total_tokens': 226, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), AIMessage(content='Transferring back to supervisor', additional_kwargs={}, response_metadata={'__is_handoff_back': True}, name='flight_assistant', id='b82588d1-cd69-4b9e-8403-bea80f909423', tool_calls=[{'name': 'transfer_back_to_supervisor', 'args': {}, 'id': '9bada7d7-3e76-4c91-a897-f770e31af9d7', 'type': 'tool_call'}]), ToolMessage(content='Successfully transferred back to supervisor', name='transfer_back_to_supervisor', id='81986903-1e5f-44c9-a024-7ed7b7289960', tool_call_id='9bada7d7-3e76-4c91-a897-f770e31af9d7'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_qogNT5MLJccSbrlLW1VnLNOM', 'function': {'arguments': '{}', 'name': 'transfer_to_hotel_assistant'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 260, 'total_tokens': 274, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_07871e2ad8', 'id': 'chatcmpl-BrEhD7HCUqv5Halb397x6GLHMu6x2', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, name='supervisor', id='run--abe9f693-71a9-4d8a-9b2b-914adddb8b50-0', tool_calls=[{'name': 'transfer_to_hotel_assistant', 'args': {}, 'id': 'call_qogNT5MLJccSbrlLW1VnLNOM', 'type': 'tool_call'}], usage_metadata={'input_tokens': 260, 'output_tokens': 14, 'total_tokens': 274, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='Successfully transferred to hotel_assistant', name='transfer_to_hotel_assistant', tool_call_id='call_qogNT5MLJccSbrlLW1VnLNOM')], 'is_last_step': False, 'remaining_steps': 24}, goto='hotel_assistant')",
#                 "exception.stacktrace": "Traceback (most recent call last):\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/opentelemetry/trace/__init__.py\", line 589, in use_span\n    yield span\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/opentelemetry/sdk/trace/__init__.py\", line 1105, in start_as_current_span\n    yield span\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/src/monocle_apptrace/instrumentation/common/wrapper.py\", line 78, in monocle_wrapper_span_processor\n    return_value = wrapped(*args, **kwargs)\n                   ^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/pregel/__init__.py\", line 2843, in invoke\n    for chunk in self.stream(\n                 ^^^^^^^^^^^^\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/pregel/__init__.py\", line 2533, in stream\n    for _ in runner.tick(\n             ^^^^^^^^^^^^\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/pregel/write.py\", line 86, in _write\n    self.do_write(\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/pregel/write.py\", line 128, in do_write\n    write(_assemble_writes(writes))\n          ^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/pregel/write.py\", line 183, in _assemble_writes\n    if ww := w.mapper(w.value):\n             ^^^^^^^^^^^^^^^^^\n  File \"/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/graph/state.py\", line 1238, in _control_branch\n    raise ParentCommand(command)\nlanggraph.errors.ParentCommand: Command(graph='supervisor:ab56bb00-b70f-2b04-c124-4741d4eeb9af', update={'messages': [HumanMessage(content='book a flight from BOS to JFK and a stay at McKittrick Hotel', additional_kwargs={}, response_metadata={}, id='dba4c19e-ccef-4b98-9e4e-f0ac699050ef'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'function': {'arguments': '{}', 'name': 'transfer_to_flight_assistant'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 100, 'total_tokens': 114, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a288987b44', 'id': 'chatcmpl-BrEh9VKqTNR9zkeavIs8VA0kKVNdp', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, name='supervisor', id='run--73f157f9-bcfd-4e1b-9cdd-e9baee185dd7-0', tool_calls=[{'name': 'transfer_to_flight_assistant', 'args': {}, 'id': 'call_AatZnRVOBzIEfrIyRoqYgYaX', 'type': 'tool_call'}], usage_metadata={'input_tokens': 100, 'output_tokens': 14, 'total_tokens': 114, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='Successfully transferred to flight_assistant', name='transfer_to_flight_assistant', id='a16d817d-3a32-42c0-a67d-fedd8b353490', tool_call_id='call_AatZnRVOBzIEfrIyRoqYgYaX'), AIMessage(content='I have successfully booked a flight for you from Boston Logan International Airport (BOS) to John F. Kennedy International Airport (JFK). Unfortunately, I cannot book a stay at McKittrick Hotel. You may want to contact the hotel directly or use a travel website for accommodation bookings. If you need further assistance, feel free to ask!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 71, 'prompt_tokens': 155, 'total_tokens': 226, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a288987b44', 'id': 'chatcmpl-BrEhBETSr4kcCM1trW7ahFHhImU44', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, name='flight_assistant', id='run--7ddb5d7f-2f9e-4446-ae88-6bcf5f56d18d-0', usage_metadata={'input_tokens': 155, 'output_tokens': 71, 'total_tokens': 226, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), AIMessage(content='Transferring back to supervisor', additional_kwargs={}, response_metadata={'__is_handoff_back': True}, name='flight_assistant', id='b82588d1-cd69-4b9e-8403-bea80f909423', tool_calls=[{'name': 'transfer_back_to_supervisor', 'args': {}, 'id': '9bada7d7-3e76-4c91-a897-f770e31af9d7', 'type': 'tool_call'}]), ToolMessage(content='Successfully transferred back to supervisor', name='transfer_back_to_supervisor', id='81986903-1e5f-44c9-a024-7ed7b7289960', tool_call_id='9bada7d7-3e76-4c91-a897-f770e31af9d7'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_qogNT5MLJccSbrlLW1VnLNOM', 'function': {'arguments': '{}', 'name': 'transfer_to_hotel_assistant'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 260, 'total_tokens': 274, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_07871e2ad8', 'id': 'chatcmpl-BrEhD7HCUqv5Halb397x6GLHMu6x2', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, name='supervisor', id='run--abe9f693-71a9-4d8a-9b2b-914adddb8b50-0', tool_calls=[{'name': 'transfer_to_hotel_assistant', 'args': {}, 'id': 'call_qogNT5MLJccSbrlLW1VnLNOM', 'type': 'tool_call'}], usage_metadata={'input_tokens': 260, 'output_tokens': 14, 'total_tokens': 274, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='Successfully transferred to hotel_assistant', name='transfer_to_hotel_assistant', tool_call_id='call_qogNT5MLJccSbrlLW1VnLNOM')], 'is_last_step': False, 'remaining_steps': 24}, goto='hotel_assistant')\n",
#                 "exception.escaped": "False"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.chat.completions.Completions.create",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x64fc17b48ee83711",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xff0217559a7b23c9",
#     "start_time": "2025-07-09T02:19:36.312858Z",
#     "end_time": "2025-07-09T02:19:37.536716Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.chat_models.base.BaseChatModel.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0xff0217559a7b23c9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7e51463fab28131a",
#     "start_time": "2025-07-09T02:19:36.311475Z",
#     "end_time": "2025-07-09T02:19:37.537613Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/runnables/base.py:5431",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o",
#         "entity.2.type": "model.llm.gpt-4o",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:37.537526Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a hotel booking assistant\"}",
#                     "{\"human\": \"book a flight from BOS to JFK and a stay at McKittrick Hotel\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully transferred to flight_assistant\"}",
#                     "{\"ai\": \"I have successfully booked a flight for you from Boston Logan International Airport (BOS) to John F. Kennedy International Airport (JFK). Unfortunately, I cannot book a stay at McKittrick Hotel. You may want to contact the hotel directly or use a travel website for accommodation bookings. If you need further assistance, feel free to ask!\"}",
#                     "{\"ai\": \"Transferring back to supervisor\"}",
#                     "{\"tool\": \"Successfully transferred back to supervisor\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully transferred to hotel_assistant\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:37.537570Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-09T02:19:37.537596Z",
#             "attributes": {
#                 "completion_tokens": 20,
#                 "prompt_tokens": 261,
#                 "total_tokens": 281
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.schema.runnable.RunnableSequence.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x7e51463fab28131a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6b5627152a1f0e4d",
#     "start_time": "2025-07-09T02:19:36.310057Z",
#     "end_time": "2025-07-09T02:19:37.537717Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/prebuilt/chat_agent_executor.py:507",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain_core.tools.base.BaseTool.run",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x4d9207b0545ad999",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6b5627152a1f0e4d",
#     "start_time": "2025-07-09T02:19:37.539818Z",
#     "end_time": "2025-07-09T02:19:37.540790Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/tools/base.py:599",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "agentic.tool",
#         "entity.1.type": "tool.langgraph",
#         "entity.1.name": "book_hotel",
#         "entity.1.description": "Book a hotel",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:37.540754Z",
#             "attributes": {
#                 "Inputs": [
#                     "McKittrick Hotel"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:37.540777Z",
#             "attributes": {
#                 "response": "Successfully booked a stay at McKittrick Hotel.",
#                 "status": "success"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.Completions.create",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x8736bf15f09956d5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x2f682616f76bead9",
#     "start_time": "2025-07-09T02:19:37.545227Z",
#     "end_time": "2025-07-09T02:19:38.333492Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.chat_models.base.BaseChatModel.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x2f682616f76bead9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x738fed32b4d3228d",
#     "start_time": "2025-07-09T02:19:37.543722Z",
#     "end_time": "2025-07-09T02:19:38.334480Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/runnables/base.py:5431",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o",
#         "entity.2.type": "model.llm.gpt-4o",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:38.334372Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a hotel booking assistant\"}",
#                     "{\"human\": \"book a flight from BOS to JFK and a stay at McKittrick Hotel\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully transferred to flight_assistant\"}",
#                     "{\"ai\": \"I have successfully booked a flight for you from Boston Logan International Airport (BOS) to John F. Kennedy International Airport (JFK). Unfortunately, I cannot book a stay at McKittrick Hotel. You may want to contact the hotel directly or use a travel website for accommodation bookings. If you need further assistance, feel free to ask!\"}",
#                     "{\"ai\": \"Transferring back to supervisor\"}",
#                     "{\"tool\": \"Successfully transferred back to supervisor\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully transferred to hotel_assistant\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully booked a stay at McKittrick Hotel.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:38.334429Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "{\"ai\": \"I have successfully booked your stay at the McKittrick Hotel. If you need anything else, feel free to let me know!\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-09T02:19:38.334458Z",
#             "attributes": {
#                 "completion_tokens": 28,
#                 "prompt_tokens": 305,
#                 "total_tokens": 333
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.schema.runnable.RunnableSequence.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x738fed32b4d3228d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6b5627152a1f0e4d",
#     "start_time": "2025-07-09T02:19:37.542712Z",
#     "end_time": "2025-07-09T02:19:38.334572Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/prebuilt/chat_agent_executor.py:507",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langgraph.graph.state.CompiledStateGraph.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x6b5627152a1f0e4d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0c7bcda7610dd38b",
#     "start_time": "2025-07-09T02:19:36.307965Z",
#     "end_time": "2025-07-09T02:19:38.335444Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph_supervisor/supervisor.py:93",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.type": "agent.langgraph",
#         "entity.1.name": "hotel_assistant",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:38.335417Z",
#             "attributes": {
#                 "query": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:38.335431Z",
#             "attributes": {
#                 "response": "I have successfully booked your stay at the McKittrick Hotel. If you need anything else, feel free to let me know!"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "openai.resources.chat.completions.Completions.create",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x2e48e9a07525e046",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x58c53968726c9491",
#     "start_time": "2025-07-09T02:19:38.341859Z",
#     "end_time": "2025-07-09T02:19:39.676422Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.chat_models.base.BaseChatModel.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x58c53968726c9491",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa4d434ab4e431409",
#     "start_time": "2025-07-09T02:19:38.340545Z",
#     "end_time": "2025-07-09T02:19:39.677166Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langchain_core/runnables/base.py:5431",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o",
#         "entity.2.type": "model.llm.gpt-4o",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:39.677074Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You manage a hotel booking assistant and aflight booking assistant. Assign work to them.\"}",
#                     "{\"human\": \"book a flight from BOS to JFK and a stay at McKittrick Hotel\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully transferred to flight_assistant\"}",
#                     "{\"ai\": \"I have successfully booked a flight for you from Boston Logan International Airport (BOS) to John F. Kennedy International Airport (JFK). Unfortunately, I cannot book a stay at McKittrick Hotel. You may want to contact the hotel directly or use a travel website for accommodation bookings. If you need further assistance, feel free to ask!\"}",
#                     "{\"ai\": \"Transferring back to supervisor\"}",
#                     "{\"tool\": \"Successfully transferred back to supervisor\"}",
#                     "{\"ai\": \"\"}",
#                     "{\"tool\": \"Successfully transferred to hotel_assistant\"}",
#                     "{\"ai\": \"I have successfully booked your stay at the McKittrick Hotel. If you need anything else, feel free to let me know!\"}",
#                     "{\"ai\": \"Transferring back to supervisor\"}",
#                     "{\"tool\": \"Successfully transferred back to supervisor\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:39.677122Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "{\"ai\": \"Your flight from Boston (BOS) to New York (JFK) has been booked, and your stay at the McKittrick Hotel is confirmed. If you need further assistance, feel free to ask!\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-09T02:19:39.677148Z",
#             "attributes": {
#                 "completion_tokens": 44,
#                 "prompt_tokens": 377,
#                 "total_tokens": 421
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langchain.schema.runnable.RunnableSequence.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0xa4d434ab4e431409",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x4282efd7f2290fa2",
#     "start_time": "2025-07-09T02:19:38.339531Z",
#     "end_time": "2025-07-09T02:19:39.677249Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/prebuilt/chat_agent_executor.py:507",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langgraph.graph.state.CompiledStateGraph.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x4282efd7f2290fa2",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0c7bcda7610dd38b",
#     "start_time": "2025-07-09T02:19:38.337073Z",
#     "end_time": "2025-07-09T02:19:39.678079Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/.env/lib/python3.12/site-packages/langgraph/utils/runnable.py:623",
#         "workflow.name": "langchain_agent_1",
#         "span.type": "agentic.invocation",
#         "entity.1.type": "agent.langgraph",
#         "entity.1.name": "supervisor",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-09T02:19:39.678055Z",
#             "attributes": {
#                 "query": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-09T02:19:39.678067Z",
#             "attributes": {
#                 "response": "Your flight from Boston (BOS) to New York (JFK) has been booked, and your stay at the McKittrick Hotel is confirmed. If you need further assistance, feel free to ask!"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "langgraph.graph.state.CompiledStateGraph.invoke",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0x0c7bcda7610dd38b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf370fcdd26a432ba",
#     "start_time": "2025-07-09T02:19:31.022088Z",
#     "end_time": "2025-07-09T02:19:39.678504Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/tests/integration/test_langgraph_multi_agent.py:68",
#         "workflow.name": "langchain_agent_1",
#         "parent.agent.span": true,
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ,{
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x517d9464829017e0ba266d3c77154837",
#         "span_id": "0xf370fcdd26a432ba",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-09T02:19:31.022039Z",
#     "end_time": "2025-07-09T02:19:39.678536Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/prasad/repos/monocle2ai/monocle-prasad/tests/integration/test_langgraph_multi_agent.py:68",
#         "span.type": "workflow",
#         "entity.1.name": "langchain_agent_1",
#         "entity.1.type": "workflow.langgraph",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_agent_1"
#         },
#         "schema_url": ""
#     }
# }
# ]