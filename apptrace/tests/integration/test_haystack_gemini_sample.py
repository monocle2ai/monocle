import logging
import os
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.generators.google_ai import (
    GoogleAIGeminiChatGenerator,
)
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="haystack_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[]
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

@pytest.mark.integration()
def test_haystack_anthropic_sample(setup):
    api_key = os.getenv("GEMINI_API_KEY")
    llm = GoogleAIGeminiChatGenerator(
        model="gemini-2.5-pro",
        api_key= Secret.from_token(api_key),
    )

    messages = [ChatMessage.from_system("Translate from English to German"),
                ChatMessage.from_user("I love programming")]

    response = llm.run(messages=messages)
    time.sleep(5)
    logger.info(response)

    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes

            assert span_attributes["entity.1.type"] == "inference.gemini"
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gemini-2.5-pro"
            assert span_attributes["entity.2.type"] == "model.llm.gemini-2.5-pro"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if not span.parent and span.name == "workflow":
            assert span_attributes["entity.1.name"] == "haystack_app_1"
            assert span_attributes["entity.1.type"] == "workflow.haystack"

# {
#     "name": "haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator",
#     "context": {
#         "trace_id": "0xde767341a275994fad1e932d932ca849",
#         "span_id": "0x37fb11ac5f5a531c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe478b9b5cf936e09",
#     "start_time": "2025-07-10T12:39:42.072349Z",
#     "end_time": "2025-07-10T12:39:44.372454Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_haystack_gemini_sample.py:30",
#         "workflow.name": "haystack_app_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.gemini",
#         "entity.1.inference_endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
#         "entity.2.name": "gemini-1.5-pro",
#         "entity.2.type": "model.llm.gemini-1.5-pro",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T12:39:44.372454Z",
#             "attributes": {
#                 "input": [
#                     "{'system': 'Translate from English to German'}",
#                     "{'user': 'I love programming'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T12:39:44.372454Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "{\"assistant\": \"Ich liebe das Programmieren.\\n\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-10T12:39:44.372454Z",
#             "attributes": {
#                 "completion_tokens": 8,
#                 "prompt_tokens": 8,
#                 "total_tokens": 16
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xde767341a275994fad1e932d932ca849",
#         "span_id": "0xe478b9b5cf936e09",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-10T12:39:42.072349Z",
#     "end_time": "2025-07-10T12:39:44.372454Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_haystack_gemini_sample.py:30",
#         "span.type": "workflow",
#         "entity.1.name": "haystack_app_1",
#         "entity.1.type": "workflow.haystack",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }