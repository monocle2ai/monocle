# Enable Monocle Tracing
import time

from monocle_apptrace.instrumentation.common.instrumentor import (
    set_context_properties,
    setup_monocle_telemetry,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
import os
from google import genai
import pytest
custom_exporter = CustomConsoleSpanExporter()
@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
                workflow_name="gemini_app_1",
                span_processors=[BatchSpanProcessor(custom_exporter)],
                wrapper_methods=[])
# Set the environment variable for the Google GenAI API keyQOP

@pytest.mark.integration()
def test_langchain_chat_sample(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Explain how AI works in a few words",
    )
    time.sleep(5)
    print(response.text)
    spans = custom_exporter.get_captured_spans()
    found_workflow_span = False
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
                span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.gemini"
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gemini-2.0-flash"
            assert span_attributes["entity.2.type"] == "model.llm.gemini-2.0-flash"

            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "workflow":
            found_workflow_span = True
    assert found_workflow_span

# {
#     "name": "google.genai.models.Models",
#     "context": {
#         "trace_id": "0x1b245918f73d03854d517f43651da341",
#         "span_id": "0x8e6d35d4972c662e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x034482a886b9e40f",
#     "start_time": "2025-06-12T07:20:38.001825Z",
#     "end_time": "2025-06-12T07:20:40.124957Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.1",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_gemini_sample.py:26",
#         "workflow.name": "gemini_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.generic",
#         "entity.1.inference_endpoint": "https://generativelanguage.googleapis.com/",
#         "entity.2.name": "gemini-2.0-flash",
#         "entity.2.type": "model.llm.gemini-2.0-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-06-12T07:20:40.124957Z",
#             "attributes": {
#                 "input": [
#                     "{'input': 'Explain how AI works in a few words'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-06-12T07:20:40.124957Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "AI learns patterns from data to make predictions or decisions.\n"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-06-12T07:20:40.124957Z",
#             "attributes": {
#                 "completion_tokens": 12,
#                 "prompt_tokens": 8,
#                 "total_tokens": 20
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x1b245918f73d03854d517f43651da341",
#         "span_id": "0x034482a886b9e40f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-06-12T07:20:38.001825Z",
#     "end_time": "2025-06-12T07:20:40.124957Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.1",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_gemini_sample.py:26",
#         "span.type": "workflow",
#         "entity.1.name": "gemini_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }