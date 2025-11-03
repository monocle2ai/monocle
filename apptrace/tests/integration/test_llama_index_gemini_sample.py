import logging
import os
import subprocess
import sys
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from llama_index.core.llms import ChatMessage
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)
@pytest.fixture(scope="module")
def setup():
    subprocess.check_call([sys.executable, "-m", "pip", "install", ".[dev_gemini]"])
    custom_exporter = CustomConsoleSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="llamaindex_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

# module cleanup function
@pytest.fixture(scope="module", autouse=True)
def cleanup_module():
    yield
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "llama-index-llms-gemini"])

def test_llamaindex_gemini_sample(setup, venv):
    # dynamically import Gemini after installing the package
    from llama_index.llms.gemini.base import Gemini
    
    llm = Gemini(
        model="gemini-2.5-pro",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
        max_output_tokens=1024,
    )
    messages = [
        ChatMessage(role="assistant", content="Translate from English to German."),
        ChatMessage(
            role="user", content="I love programming"
        ),
    ]

    ai_answer = llm.chat(messages)
    logger.info(ai_answer)
    time.sleep(5)
    found_workflow_span = False
    spans = setup.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        if "span.type" in span_attributes and (span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            assert span_attributes["entity.1.type"] == "inference.gemini"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gemini-2.5-pro"
            assert span_attributes["entity.2.type"] == "model.llm.gemini-2.5-pro"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events

            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "workflow":
            found_workflow_span = True
    assert found_workflow_span

# {
#     "name": "llama_index.llms.gemini.base.Gemini",
#     "context": {
#         "trace_id": "0x6ca308e92e8aa53edbefacc3a30e653f",
#         "span_id": "0x21cdaf43569b6041",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc8dc4870007af0f3",
#     "start_time": "2025-07-10T09:40:25.499564Z",
#     "end_time": "2025-07-10T09:42:55.784706Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_llama_index_gemini_sample.py:36",
#         "workflow.name": "llamaindex_app_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.gemini",
#         "entity.1.provider_name": "gemini.googleapis.com",
#         "entity.1.inference_endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
#         "entity.2.name": "gemini-1.5-flash",
#         "entity.2.type": "model.llm.gemini-1.5-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-10T09:40:43.416607Z",
#             "attributes": {
#                 "input": [
#                     "{\"assistant\": \"Translate from English to German.\"}",
#                     "{\"user\": \"I love programming\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-10T09:40:43.416607Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "{\"assistant\": \"Ich liebe Programmieren.\\n\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-10T09:42:55.783568Z",
#             "attributes": {
#                 "temperature": 0.0,
#                 "completion_tokens": 7,
#                 "prompt_tokens": 9,
#                 "total_tokens": 16
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_app_1"
#         },
#         "schema_url": ""
#     }
# }
#
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x6ca308e92e8aa53edbefacc3a30e653f",
#         "span_id": "0xc8dc4870007af0f3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-10T09:40:25.499564Z",
#     "end_time": "2025-07-10T09:42:55.784706Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_llama_index_gemini_sample.py:36",
#         "span.type": "workflow",
#         "entity.1.name": "llamaindex_app_1",
#         "entity.1.type": "workflow.llamaindex",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llamaindex_app_1"
#         },
#         "schema_url": ""
#     }
# }