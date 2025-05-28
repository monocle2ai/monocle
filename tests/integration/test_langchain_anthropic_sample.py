import time
import os
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
                workflow_name="langchain_app_1",
                span_processors=[BatchSpanProcessor(custom_exporter)],
                wrapper_methods=[])

@pytest.mark.integration()
def test_langchain_anthropic_sample(setup):
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    ai_answer = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    time.sleep(5)
    print(ai_answer)

    spans = custom_exporter.get_captured_spans()

    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes

            assert span_attributes["entity.1.type"] == "inference.anthropic"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "claude-3-5-sonnet-20240620"
            assert span_attributes["entity.2.type"] == "model.llm.claude-3-5-sonnet-20240620"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if not span.parent and span.name == "workflow":
            assert span_attributes["entity.1.name"] == "langchain_app_1"
            assert span_attributes["entity.1.type"] == "workflow.langchain"

# {
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x86b8f96576496c82450092a3cf9e67de",
#         "span_id": "0x1bbcac2eda372cff",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6edea8aa19f76d39",
#     "start_time": "2025-04-17T07:52:39.528309Z",
#     "end_time": "2025-04-17T07:52:39.529308Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "anthropic.resources.messages.messages.Messages",
#     "context": {
#         "trace_id": "0x86b8f96576496c82450092a3cf9e67de",
#         "span_id": "0xdd04199b4877d905",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xbcda108f537ad78f",
#     "start_time": "2025-04-17T07:52:39.537308Z",
#     "end_time": "2025-04-17T07:52:41.347488Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_anthropic.chat_models.ChatAnthropic",
#     "context": {
#         "trace_id": "0x86b8f96576496c82450092a3cf9e67de",
#         "span_id": "0xbcda108f537ad78f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6edea8aa19f76d39",
#     "start_time": "2025-04-17T07:52:39.529308Z",
#     "end_time": "2025-04-17T07:52:41.351488Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.anthropic",
#         "entity.1.provider_name": "api.anthropic.com",
#         "entity.1.inference_endpoint": "https://api.anthropic.com",
#         "entity.2.name": "claude-3-5-sonnet-20240620",
#         "entity.2.type": "model.llm.claude-3-5-sonnet-20240620",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-04-17T07:52:41.351488Z",
#             "attributes": {
#                 "input": [
#                     "{'system': 'You are a helpful assistant that translates English to German.'}",
#                     "{'human': 'I love programming.'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-04-17T07:52:41.351488Z",
#             "attributes": {
#                 "response": [
#                     "Here's the German translation:\n\nIch liebe Programmieren."
#                 ]
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-04-17T07:52:41.351488Z",
#             "attributes": {
#                 "temperature": 0.0,
#                 "completion_tokens": 18,
#                 "prompt_tokens": 23,
#                 "total_tokens": 41
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.runnables.base.RunnableSequence",
#     "context": {
#         "trace_id": "0x86b8f96576496c82450092a3cf9e67de",
#         "span_id": "0x6edea8aa19f76d39",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf74bceb6322023b6",
#     "start_time": "2025-04-17T07:52:39.514040Z",
#     "end_time": "2025-04-17T07:52:41.351488Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x86b8f96576496c82450092a3cf9e67de",
#         "span_id": "0xf74bceb6322023b6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-04-17T07:52:39.514040Z",
#     "end_time": "2025-04-17T07:52:41.351488Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "span.type": "workflow",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.langchain",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }