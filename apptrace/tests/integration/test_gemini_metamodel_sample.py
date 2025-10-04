# Enable Monocle Tracing
import logging
import os
import time

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
            workflow_name="gemini_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_gemini_model_sample(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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


def test_gemini_chat_sample(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    chat = client.chats.create(model="gemini-2.5-flash")

    response = chat.send_message("I have 2 dogs in my house.")
    logger.info(response.text)

    response = chat.send_message("How many paws are in my house?")
    logger.info(response.text)

    for message in chat.get_history():
        logger.info(f"role - {message.role}: {message.parts[0].text}")
    time.sleep(5)
    logger.info(response.text)
    spans = setup.get_captured_spans()
    check_span_chat(spans)


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
            entity_type="inference.gemini",
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


def check_span_chat(spans):
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
            entity_type="inference.gemini",
            model_name="gemini-2.5-flash",
            model_type="model.llm.gemini-2.5-flash",
            check_metadata=True,
            check_input_output=True,
        )
    assert (
        len(inference_spans) == 2
    ), "Expected exactly two inference spans for the LLM call"

    for span in inference_spans:
        # Validate events using the generic function with regex patterns
        if len(span.events[0].attributes.get("input")) == 1:
            # This is the first message with user input
            input_patterns = [r"^\{\"user\": \".+\"\}$"]  # Pattern for user message
        else:
            # This is the second message with user input and AI response
            input_patterns = [
                r"^\{\"user\": \".+\"\}$",  # Pattern for user message
                r"^\{\"model\": \".+\"\}$",  # Pattern for AI response
                r"^\{\"user\": \".+\"\}$",  # Pattern for user message
            ]
        validate_inference_span_events(
            span=span,
            expected_event_count=3,
            input_patterns=input_patterns,
            # TODO fix all outputs and make sure that we dont use python str
            # then we can uncomment this line
            # output_pattern=r"^\{'model': '.+'\}$",  # Pattern for AI response
            metadata_requirements={
                "completion_tokens": int,
                "prompt_tokens": int,
                "total_tokens": int,
            },
        )


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "google.genai.models.Models",
#     "context": {
#         "trace_id": "0x51b8d7a9d3c6cdb70a8be5be8cfb6734",
#         "span_id": "0xd35571f37f9174bb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6ac6f794cf706c3d",
#     "start_time": "2025-07-13T11:21:13.024409Z",
#     "end_time": "2025-07-13T11:21:17.993373Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_gemini_metamodel_sample.py:38",
#         "workflow.name": "gemini_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.gemini",
#         "entity.1.inference_endpoint": "https://generativelanguage.googleapis.com/",
#         "entity.2.name": "gemini-2.5-flash",
#         "entity.2.type": "model.llm.gemini-2.5-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-13T11:21:17.993309Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a cat. Your name is Neko.\"}",
#                     "{\"user\": \"Hello there\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-13T11:21:17.993349Z",
#             "attributes": {
#                 "response": "{\"model\": \"Mrow. *I tilt my head slightly, my ears swiveling to catch your voice, then give a slow, deliberate blink before my tail twitches just once.*\"}",
#                 "finish_reason": "STOP",
#                 "finish_type": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-13T11:21:17.993360Z",
#             "attributes": {
#                 "completion_tokens": 35,
#                 "prompt_tokens": 15,
#                 "total_tokens": 432
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
#         "trace_id": "0x51b8d7a9d3c6cdb70a8be5be8cfb6734",
#         "span_id": "0x6ac6f794cf706c3d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-13T11:21:13.024349Z",
#     "end_time": "2025-07-13T11:21:17.993388Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_gemini_metamodel_sample.py:38",
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
# {
#     "name": "google.genai.models.Models",
#     "context": {
#         "trace_id": "0x9ef3160d23ffb494cc5cd0459130abd0",
#         "span_id": "0x4bb294950c801e61",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0bff0439ce2d9dde",
#     "start_time": "2025-07-13T11:21:23.022167Z",
#     "end_time": "2025-07-13T11:21:29.499246Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/google/genai/chats.py:254",
#         "workflow.name": "gemini_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.gemini",
#         "entity.1.inference_endpoint": "https://generativelanguage.googleapis.com/",
#         "entity.2.name": "gemini-2.5-flash",
#         "entity.2.type": "model.llm.gemini-2.5-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-13T11:21:29.499101Z",
#             "attributes": {
#                 "input": [
#                     "{\"user\": \"I have 2 dogs in my house.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-13T11:21:29.499202Z",
#             "attributes": {
#                 "response": "{\"model\": \"Oh, how wonderful! Two dogs must bring a lot of joy (and maybe a little playful chaos!) to your home.\\n\\nDo you want to tell me anything about them? Like their names, breeds, or what they're like?\"}",
#                 "finish_reason": "STOP",
#                 "finish_type": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-13T11:21:29.499225Z",
#             "attributes": {
#                 "completion_tokens": 49,
#                 "prompt_tokens": 10,
#                 "total_tokens": 1030
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
#         "trace_id": "0x9ef3160d23ffb494cc5cd0459130abd0",
#         "span_id": "0x0bff0439ce2d9dde",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-13T11:21:23.021999Z",
#     "end_time": "2025-07-13T11:21:29.499277Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/google/genai/chats.py:254",
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
# {
#     "name": "google.genai.models.Models",
#     "context": {
#         "trace_id": "0x1f61b158d782c2461e0b7a9c03f703a4",
#         "span_id": "0x56149ede73636465",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa9c42e2a6fb1c32d",
#     "start_time": "2025-07-13T11:21:29.499967Z",
#     "end_time": "2025-07-13T11:21:33.796859Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/google/genai/chats.py:254",
#         "workflow.name": "gemini_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.gemini",
#         "entity.1.inference_endpoint": "https://generativelanguage.googleapis.com/",
#         "entity.2.name": "gemini-2.5-flash",
#         "entity.2.type": "model.llm.gemini-2.5-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-13T11:21:33.796812Z",
#             "attributes": {
#                 "input": [
#                     "{\"user\": \"I have 2 dogs in my house.\"}",
#                     "{\"model\": \"Oh, how wonderful! Two dogs must bring a lot of joy (and maybe a little playful chaos!) to your home.\\n\\nDo you want to tell me anything about them? Like their names, breeds, or what they're like?\"}",
#                     "{\"user\": \"How many paws are in my house?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-13T11:21:33.796842Z",
#             "attributes": {
#                 "response": "{\"model\": \"That's a fun question!\\n\\nAssuming your two dogs each have the standard four paws:\\n\\n2 dogs x 4 paws/dog = **8 paws**\\n\\nOf course, you also have 2 feet, which aren't typically called paws, but are certainly \\\"foot-shaped\\\"! So depending on how broadly you're counting, it could be 8 or 10.\"}",
#                 "finish_reason": "STOP",
#                 "finish_type": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-13T11:21:33.796850Z",
#             "attributes": {
#                 "completion_tokens": 80,
#                 "prompt_tokens": 69,
#                 "total_tokens": 849
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
#         "trace_id": "0x1f61b158d782c2461e0b7a9c03f703a4",
#         "span_id": "0xa9c42e2a6fb1c32d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-13T11:21:29.499865Z",
#     "end_time": "2025-07-13T11:21:33.796870Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/google/genai/chats.py:254",
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