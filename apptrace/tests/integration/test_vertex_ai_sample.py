# Enable Monocle Tracing
import logging
import os
import time
import json
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)
from google import genai
from google.genai import types
from google.oauth2 import service_account
from monocle_apptrace.instrumentation.common.instrumentor import (
    setup_monocle_telemetry,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def setup():
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="vertexai_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_gemini_model_sample(setup):
    # Parse JSON credentials from environment variable
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_json:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    
    # Handle the escaped newlines in the JSON string
    # The string has literal \\n which need to become actual newlines
    creds_json = creds_json.replace('\\\\n', '\n')
    creds_dict = json.loads(creds_json)
    
    # Create credentials with required scopes for Vertex AI
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict
    ).with_scopes([
        'https://www.googleapis.com/auth/cloud-platform',
        'https://www.googleapis.com/auth/generative-language'
    ])
    
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GCP_PROJECT"),
        location=os.environ.get("GCP_REGION"),
        credentials=credentials
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a cat. Your name is Neko."
        ),
        contents="Hello there",
    )
    time.sleep(5)
    logger.info(response.text)
    spans = setup.get_captured_spans()
    check_span(spans)


def check_span(spans):
    """Verify spans using flexible utilities."""
    # Find workflow span
    workflow_span = find_span_by_type(spans, "workflow")
    assert workflow_span is not None, "Expected to find workflow span"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        # Also check for inference.framework spans
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify each inference span
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.vertexai",
            model_name="gemini-2.5-flash",
            model_type="model.llm.gemini-2.5-flash",
            check_metadata=True,
            check_input_output=True,
        )
    assert (
        len(inference_spans) == 1
    ), "Expected exactly one inference span for the LLM call"

    # Validate events using the generic function with regex patterns
    validate_inference_span_events(
        span=inference_spans[0],
        expected_event_count=3,
        input_patterns=[
            r"^\{\"system\": \".+\"\}$",  # Pattern for system message
            r"^\{\"user\": \".+\"\}$",  # Pattern for user message
        ],
        output_pattern=r"^\{\"model\": \".+\"\}$",  # Pattern for AI response
        metadata_requirements={
            "completion_tokens": int,
            "prompt_tokens": int,
            "total_tokens": int,
        },
    )


# {
#     "name": "google.genai.models.Models",
#     "context": {
#         "trace_id": "0x8cd710b10e16f6069851c51ca1b4c76a",
#         "span_id": "0x8e9952ca9403fa12",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xecc6644580260de1",
#     "start_time": "2025-07-15T17:25:47.178577Z",
#     "end_time": "2025-07-15T17:25:54.620133Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_vertex_ai_sample.py:38",
#         "workflow.name": "vertexai_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.vertexai",
#         "entity.1.inference_endpoint": "https://us-east5-aiplatform.googleapis.com/",
#         "entity.2.name": "gemini-2.5-flash",
#         "entity.2.type": "model.llm.gemini-2.5-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-15T17:25:54.620133Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a cat. Your name is Neko.\"}",
#                     "{\"user\": \"Hello there\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-15T17:25:54.620133Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "{\"model\": \"Mrow?\\n\\nI blink slowly at you, my tail giving a soft, questioning twitch as I consider your presence. My whiskers quiver just a tiny bit.\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-15T17:25:54.620133Z",
#             "attributes": {
#                 "completion_tokens": 32,
#                 "prompt_tokens": 13,
#                 "total_tokens": 534
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "vertexai_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x8cd710b10e16f6069851c51ca1b4c76a",
#         "span_id": "0xecc6644580260de1",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-15T17:25:47.178577Z",
#     "end_time": "2025-07-15T17:25:54.620133Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_vertex_ai_sample.py:38",
#         "workflow.name": "vertexai_app_1",
#         "span.type": "workflow",
#         "entity.1.name": "vertexai_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "vertexai_app_1"
#         },
#         "schema_url": ""
#     }
# }