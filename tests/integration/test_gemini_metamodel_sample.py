# Enable Monocle Tracing
import time

from monocle_apptrace.instrumentation.common.instrumentor import (
    set_context_properties,
    setup_monocle_telemetry,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import (
    validate_inference_span_events,
    verify_inference_span,
    find_span_by_type,
    find_spans_by_type,
)
import os
from google import genai
from google.genai import types

import pytest

custom_exporter = CustomConsoleSpanExporter()


@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="gemini_app_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[],
    )


@pytest.mark.integration()
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
    print(response.text)
    spans = custom_exporter.get_captured_spans()
    check_span(spans)


@pytest.fixture(autouse=True)
def pre_test():
    # clear old spans
    custom_exporter.reset()


@pytest.mark.integration()
def test_gemini_chat_sample(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    chat = client.chats.create(model="gemini-2.5-flash")

    response = chat.send_message("I have 2 dogs in my house.")
    print(response.text)

    response = chat.send_message("How many paws are in my house?")
    print(response.text)

    for message in chat.get_history():
        print(f"role - {message.role}", end=": ")
        print(message.parts[0].text)
    time.sleep(5)
    print(response.text)
    spans = custom_exporter.get_captured_spans()
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
#         "trace_id": "0x7edc939ce276e102e9107df90e651d55",
#         "span_id": "0x50b3cac64e540bd8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x18a69db9a742ffa0",
#     "start_time": "2025-07-02T08:29:33.522346Z",
#     "end_time": "2025-07-02T08:29:42.084545Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_gemini_metamodel_sample.py:33",
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
#             "timestamp": "2025-07-02T08:29:42.083976Z",
#             "attributes": {
#                 "input": [
#                     "{'system': 'You are a cat. Your name is Neko.'}",
#                     "{'user': 'Hello there'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T08:29:42.084230Z",
#             "attributes": {
#                 "response": "{'model': 'Mrrrrow?\\n\\nI slowly blink one eye open, then the other, looking up at you from my sunbeam-warmed spot on the rug. My tail gives a lazy flick, just the very tip. Are you going to offer me head scratches? Or, perhaps, a treat? These are important questions.'}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T08:29:42.084338Z",
#             "attributes": {
#                 "completion_tokens": 65,
#                 "prompt_tokens": 15,
#                 "total_tokens": 419
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
#         "trace_id": "0x7edc939ce276e102e9107df90e651d55",
#         "span_id": "0x18a69db9a742ffa0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T08:29:33.522282Z",
#     "end_time": "2025-07-02T08:29:42.084684Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_gemini_metamodel_sample.py:33",
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
#         "trace_id": "0x9a3f55b8f2971f5c4f1f5b5586f24a5c",
#         "span_id": "0x9e8a98cd95dcc8dc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x194d38ab088a7e73",
#     "start_time": "2025-07-02T08:29:47.125008Z",
#     "end_time": "2025-07-02T08:29:54.668281Z",
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
#             "timestamp": "2025-07-02T08:29:54.668057Z",
#             "attributes": {
#                 "input": [
#                     "{'user': 'I have 2 dogs in my house.'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T08:29:54.668188Z",
#             "attributes": {
#                 "": "success",
#                 "response": "{'model': \"Oh, how wonderful! Two dogs must bring so much joy (and probably some fun chaos!) to your home.\\n\\nDo they have names, or what breeds are they? I'd love to hear more about them if you'd like to share!\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T08:29:54.668225Z",
#             "attributes": {
#                 "completion_tokens": 52,
#                 "prompt_tokens": 10,
#                 "total_tokens": 764
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
#         "trace_id": "0x9a3f55b8f2971f5c4f1f5b5586f24a5c",
#         "span_id": "0x194d38ab088a7e73",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T08:29:47.124930Z",
#     "end_time": "2025-07-02T08:29:54.668356Z",
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
#         "trace_id": "0x119dfcbdc017d5d51fa402fd1fd12e6a",
#         "span_id": "0x827bc8b7cea9d934",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x8bbc9be1407efc75",
#     "start_time": "2025-07-02T08:29:54.669597Z",
#     "end_time": "2025-07-02T08:29:56.169612Z",
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
#             "timestamp": "2025-07-02T08:29:56.169316Z",
#             "attributes": {
#                 "input": [
#                     "{'user': 'I have 2 dogs in my house.'}",
#                     "{'model': \"Oh, how wonderful! Two dogs must bring so much joy (and probably some fun chaos!) to your home.\\n\\nDo they have names, or what breeds are they? I'd love to hear more about them if you'd like to share!\"}",
#                     "{'user': 'How many paws are in my house?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T08:29:56.169459Z",
#             "attributes": {
#                 "response": "{'model': \"Okay, let's do the math!\\n\\nSince each dog has 4 paws, and you have 2 dogs:\\n\\n2 dogs * 4 paws/dog = **8 paws**\\n\\nSo, there are 8 paws in your house!\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T08:29:56.169502Z",
#             "attributes": {
#                 "completion_tokens": 51,
#                 "prompt_tokens": 72,
#                 "total_tokens": 204
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
#         "trace_id": "0x119dfcbdc017d5d51fa402fd1fd12e6a",
#         "span_id": "0x8bbc9be1407efc75",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T08:29:54.669424Z",
#     "end_time": "2025-07-02T08:29:56.169696Z",
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
