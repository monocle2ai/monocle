import logging
import os
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="langchain_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def test_langchain_gemini_sample(setup):
    os.environ.setdefault("GOOGLE_API_KEY", "GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        max_output_tokens=1024,
        google_api_key=os.getenv("GEMINI_API_KEY"),
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
    logger.info(ai_answer)
    time.sleep(5)
    found_workflow_span = False
    spans = setup.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        if "span.type" in span_attributes and (
                span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.gemini"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gemini-2.5-pro"
            assert span_attributes["entity.2.type"] == "model.llm.models/gemini-2.5-pro"

            span_input, span_output, span_metadata = span.events

            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "workflow":
            found_workflow_span = True
    assert found_workflow_span


# {
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x2bf66a8ffc5f3219589a0a2f3a4ef028",
#         "span_id": "0xadc7925229749b6e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x4d97f986690e9202",
#     "start_time": "2025-07-10T07:05:21.317137Z",
#     "end_time": "2025-07-10T07:05:21.318137Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\checkvenv\\Lib\\site-packages\\langchain_core\\runnables\\base.py:3045",
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
#     "name": "langchain_google_genai.chat_models.ChatGoogleGenerativeAI",
#     "context": {
#         "trace_id": "0x2bf66a8ffc5f3219589a0a2f3a4ef028",
#         "span_id": "0xcd105da666933723",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x4d97f986690e9202",
#     "start_time": "2025-07-10T07:05:21.320155Z",
#     "end_time": "2025-07-10T07:05:23.796371Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\checkvenv\\Lib\\site-packages\\langchain_google_genai\\chat_models.py:1334",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.gemini",
#         "entity.1.provider_name": "googleapis.com",
#         "entity.1.inference_endpoint": "generativelanguage.googleapis.com:443",
#         "entity.2.name": "models/gemini-1.5-flash",
#         "entity.2.type": "model.llm.models/gemini-1.5-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T07:05:23.796371Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a helpful assistant that translates English to German.\"}",
#                     "{\"human\": \"I love programming.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T07:05:23.796371Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "{\"ai\": \"Ich liebe Programmieren.\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-10T07:05:23.796371Z",
#             "attributes": {
#                 "temperature": 0.0,
#                 "completion_tokens": 7,
#                 "prompt_tokens": 15,
#                 "total_tokens": 22
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
#         "trace_id": "0x2bf66a8ffc5f3219589a0a2f3a4ef028",
#         "span_id": "0x4d97f986690e9202",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc3217219b929f008",
#     "start_time": "2025-07-10T07:05:21.305769Z",
#     "end_time": "2025-07-10T07:05:23.796371Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_langchain_gemini_sample.py:41",
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
#         "trace_id": "0x2bf66a8ffc5f3219589a0a2f3a4ef028",
#         "span_id": "0xc3217219b929f008",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-10T07:05:21.305769Z",
#     "end_time": "2025-07-10T07:05:23.796371Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_langchain_gemini_sample.py:41",
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