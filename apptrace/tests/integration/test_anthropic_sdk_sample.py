import logging
import time

import anthropic
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.utils import logger
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup():
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="anthropic_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[
            ])
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def test_anthropic_metamodel_sample(setup):
    client = anthropic.Anthropic()

    # Send a prompt to Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",  # You can use claude-3-haiku, claude-3-sonnet, etc.
        max_tokens=512,
        temperature=0.7,
        system= "You are a helpful assistant to answer questions about coffee.",
        messages=[
            {"role": "user", "content": "What is an americano?"}
        ]
    )

    # Print the response
    logger.info("Claude's response:\n")
    logger.info(response.content[0].text)

    time.sleep(5)
    spans = setup.get_captured_spans()


    assert len(spans) > 0, "No spans captured for the LangChain Anthropic sample"

    workflow_span = None

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        # Also check for inference.framework spans
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify each inference span
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.anthropic",
            model_name="claude-3-5-sonnet-20240620",
            model_type="model.llm.claude-3-5-sonnet-20240620",
            check_metadata=True,
            check_input_output=True,
        )
        # Add assertion for span.subtype
        assert "span.subtype" in span.attributes, "Expected span.subtype attribute to be present"
        assert span.attributes.get("span.subtype") in ["turn_end", "tool_call", "delegation"]

    assert (
        len(inference_spans) == 1
    ), "Expected exactly one inference span for the LLM call"
    
    # Validate events using the generic function with regex patterns
    validate_inference_span_events(
        span=inference_spans[0],
        expected_event_count=3,
        input_patterns=[
            r"^\{\"system\": \".+\"\}$",  # Pattern for system
            r"^\{\"user\": \".+\"\}$",  # Pattern for user input
        ],
        output_pattern=r"^\{\"assistant\": \".+\"\}$",  # Pattern for AI response
        metadata_requirements={
            "completion_tokens": int,
            "prompt_tokens": int,
            "total_tokens": int
        }
    )
    
    workflow_span = find_span_by_type(spans, "workflow")
    
    assert workflow_span is not None, "Expected to find workflow span"
    
    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "anthropic_app_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.anthropic"

def test_anthropic_invalid_api_key(setup):
    try:
        client = anthropic.Anthropic(api_key="invalid_key_123")
        response = client.messages.create(
            model="claude-3-sonnet-20240620",
            max_tokens=512,
            system="You are a helpful assistant to answer questions about coffee.",
            messages=[
                {"role": "user", "content": "What is an americano?"}
            ]

        )
    except anthropic.APIError as e:
        logger.info("Authentication error: %s", str(e))
        assert e.status_code == 401

    time.sleep(5)
    spans = setup.get_captured_spans()
    for span in spans:
        if span.attributes.get("span.type") == "inference" or span.attributes.get("span.type") == "inference.framework":
            events = [e for e in span.events if e.name == "data.output"]
            assert len(events) > 0
            assert span.status.status_code.value == 2  # ERROR status code
            assert events[0].attributes["error_code"] == "error"
            assert "error_code" in events[0].attributes
            assert "authentication_error" in events[0].attributes.get("response", "").lower()

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "anthropic.resources.messages.messages.Messages",
#     "context": {
#         "trace_id": "0xa4f952152900a57c15968c339127175c",
#         "span_id": "0x24defb2fe4c0f83b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc0726ead9ef7cfb1",
#     "start_time": "2025-07-02T14:46:12.975954Z",
#     "end_time": "2025-07-02T14:46:19.319905Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_anthropic_sdk_sample.py:31",
#         "workflow.name": "anthropic_app_1",
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
#             "timestamp": "2025-07-02T14:46:19.319781Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a helpful assistant to answer questions about coffee.\"}",
#                     "{\"user\": \"What is an americano?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T14:46:19.319832Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"An Americano is a popular coffee drink. \"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T14:46:19.319855Z",
#             "attributes": {
#                 "completion_tokens": 288,
#                 "prompt_tokens": 24,
#                 "total_tokens": 312
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "anthropic_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xa4f952152900a57c15968c339127175c",
#         "span_id": "0xc0726ead9ef7cfb1",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T14:46:12.975899Z",
#     "end_time": "2025-07-02T14:46:19.319952Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_anthropic_sdk_sample.py:31",
#         "span.type": "workflow",
#         "entity.1.name": "anthropic_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "anthropic_app_1"
#         },
#         "schema_url": ""
#     }
# }
# F{
#     "name": "anthropic.resources.messages.messages.Messages",
#     "context": {
#         "trace_id": "0x6765541e4dd7f7cd2fa69c1765a77b3a",
#         "span_id": "0x97c91c8db5330c3c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x584454580931341d",
#     "start_time": "2025-07-02T14:46:24.347372Z",
#     "end_time": "2025-07-02T14:46:24.770248Z",
#     "status": {
#         "status_code": "ERROR",
#         "description": "AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_anthropic_sdk_sample.py:102",
#         "workflow.name": "anthropic_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.anthropic",
#         "entity.1.provider_name": "api.anthropic.com",
#         "entity.1.inference_endpoint": "https://api.anthropic.com",
#         "entity.2.name": "claude-3-sonnet-20240620",
#         "entity.2.type": "model.llm.claude-3-sonnet-20240620",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-02T14:46:24.764332Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a helpful assistant to answer questions about coffee.\"}",
#                     "{\"user\": \"What is an americano?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T14:46:24.764380Z",
#             "attributes": {
#                 "error_code": 401,
#                 "response": "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T14:46:24.764393Z",
#             "attributes": {}
#         },
#         {
#             "name": "exception",
#             "timestamp": "2025-07-02T14:46:24.770215Z",
#             "attributes": {
#                 "exception.type": "anthropic.AuthenticationError",
#                 "exception.message": "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}",
#                 "exception.stacktrace": "Traceback (most recent call last):\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/opentelemetry/trace/__init__.py\", line 587, in use_span\n    yield span\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/opentelemetry/sdk/trace/__init__.py\", line 1105, in start_as_current_span\n    yield span\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/src/monocle_apptrace/instrumentation/common/wrapper.py\", line 78, in monocle_wrapper_span_processor\n    return_value = wrapped(*args, **kwargs)\n                   ^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/anthropic/_utils/_utils.py\", line 283, in wrapper\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/anthropic/resources/messages/messages.py\", line 978, in create\n    return self._post(\n           ^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/anthropic/_base_client.py\", line 1290, in post\n    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))\n                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/anthropic/_base_client.py\", line 1085, in request\n    raise self._make_status_error_from_response(err.response) from None\nanthropic.AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}\n",
#                 "exception.escaped": "False"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "anthropic_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x6765541e4dd7f7cd2fa69c1765a77b3a",
#         "span_id": "0x584454580931341d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T14:46:24.347260Z",
#     "end_time": "2025-07-02T14:46:24.771251Z",
#     "status": {
#         "status_code": "ERROR",
#         "description": "AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_anthropic_sdk_sample.py:102",
#         "span.type": "workflow",
#         "entity.1.name": "anthropic_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [
#         {
#             "name": "exception",
#             "timestamp": "2025-07-02T14:46:24.771232Z",
#             "attributes": {
#                 "exception.type": "anthropic.AuthenticationError",
#                 "exception.message": "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}",
#                 "exception.stacktrace": "Traceback (most recent call last):\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/opentelemetry/trace/__init__.py\", line 587, in use_span\n    yield span\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/opentelemetry/sdk/trace/__init__.py\", line 1105, in start_as_current_span\n    yield span\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/src/monocle_apptrace/instrumentation/common/wrapper.py\", line 70, in monocle_wrapper_span_processor\n    return_value, span_status = monocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, False, args, kwargs)\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/src/monocle_apptrace/instrumentation/common/wrapper.py\", line 78, in monocle_wrapper_span_processor\n    return_value = wrapped(*args, **kwargs)\n                   ^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/anthropic/_utils/_utils.py\", line 283, in wrapper\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/anthropic/resources/messages/messages.py\", line 978, in create\n    return self._post(\n           ^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/anthropic/_base_client.py\", line 1290, in post\n    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))\n                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/anthropic/_base_client.py\", line 1085, in request\n    raise self._make_status_error_from_response(err.response) from None\nanthropic.AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}\n",
#                 "exception.escaped": "False"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "anthropic_app_1"
#         },
#         "schema_url": ""
#     }
# }