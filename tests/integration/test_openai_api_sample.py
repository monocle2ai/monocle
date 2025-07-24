import os
import time
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.utils import logger
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from openai import OpenAI, OpenAIError

from tests.common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)

custom_exporter = CustomConsoleSpanExporter()


@pytest.fixture(autouse=True)
def clear_spans():
    """Clear spans before each test"""
    custom_exporter.reset()
    yield


@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="generic_openai_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[],
    )


@pytest.mark.integration()
def test_openai_api_sample(setup):
    openai = OpenAI()
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )
    time.sleep(5)

    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")
    
    # Verify we have spans
    assert len(spans) > 0, "No spans captured"

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
            entity_type="inference.openai",
            model_name="gpt-4o-mini",
            model_type="model.llm.gpt-4o-mini",
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
    assert workflow_span.attributes["entity.1.name"] == "generic_openai_1"
    assert workflow_span.attributes["workflow.name"] == "generic_openai_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.openai"


@pytest.mark.integration()
def test_openai_invalid_api_key(setup):
    try:
        client = OpenAI(api_key="invalid_key_123")
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "test"}]
        )
    except OpenAIError as e:
        logger.error("Authentication error: %s", str(e))

    time.sleep(5)
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        if (
            span.attributes.get("span.type") == "inference"
            or span.attributes.get("span.type") == "inference.framework"
        ):
            events = [e for e in span.events if e.name == "data.output"]
            assert len(events) > 0
            assert span.status.status_code.value == 2  # ERROR status code
            assert events[0].attributes["error_code"] == "invalid_api_key"
            assert "error code: 401" in events[0].attributes.get("response", "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
    # Make sure to set the OPENAI_API_KEY environment variable if needed.


# {
#     "name": "openai.resources.chat.completions.completions.Completions",
#     "context": {
#         "trace_id": "0xc6deee5174f0412860594d4719fb4088",
#         "span_id": "0x1994414be3d9c0fc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe3b860fc2e5ed4f0",
#     "start_time": "2025-07-03T00:01:48.007044Z",
#     "end_time": "2025-07-03T00:01:51.483851Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample.py:39",
#         "workflow.name": "generic_openai_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-03T00:01:51.483660Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a helpful assistant to answer coffee related questions\"}",
#                     "{\"user\": \"What is an americano?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-03T00:01:51.483746Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"An Americano is a type of coffee drink that consists of espresso diluted with hot water. \"}",
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-03T00:01:51.483778Z",
#             "attributes": {
#                 "completion_tokens": 128,
#                 "prompt_tokens": 26,
#                 "total_tokens": 154
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "generic_openai_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xc6deee5174f0412860594d4719fb4088",
#         "span_id": "0xe3b860fc2e5ed4f0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-03T00:01:48.006990Z",
#     "end_time": "2025-07-03T00:01:51.483923Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample.py:39",
#         "span.type": "workflow",
#         "workflow.name": "generic_openai_1",
#         "entity.1.name": "generic_openai_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "generic_openai_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.chat.completions.completions.Completions",
#     "context": {
#         "trace_id": "0xb5892f649dbc612a67ccc496b85d9a6a",
#         "span_id": "0x5d6e6c73b00d7cbb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe59fb561d29eafcf",
#     "start_time": "2025-07-03T00:01:56.506465Z",
#     "end_time": "2025-07-03T00:01:56.880906Z",
#     "status": {
#         "status_code": "ERROR",
#         "description": "AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: invalid_***_123. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample.py:109",
#         "workflow.name": "generic_openai_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-03T00:01:56.872922Z",
#             "attributes": {
#                 "input": [
#                     "{'user': 'test'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-03T00:01:56.872976Z",
#             "attributes": {
#                 "response": "Error code: 401 - {'error': {'message': 'Incorrect API key provided: invalid_***_123. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}",
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-03T00:01:56.872992Z",
#             "attributes": {}
#         },
#         {
#             "name": "exception",
#             "timestamp": "2025-07-03T00:01:56.880870Z",
#             "attributes": {
#                 "exception.type": "openai.AuthenticationError",
#                 "exception.message": "Error code: 401 - {'error': {'message': 'Incorrect API key provided: invalid_***_123. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}",
#                 "exception.stacktrace": "Traceback (most recent call last):\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/opentelemetry/trace/__init__.py\", line 587, in use_span\n    yield span\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/opentelemetry/sdk/trace/__init__.py\", line 1105, in start_as_current_span\n    yield span\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/src/monocle_apptrace/instrumentation/common/wrapper.py\", line 78, in monocle_wrapper_span_processor\n    return_value = wrapped(*args, **kwargs)\n                   ^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/openai/_utils/_utils.py\", line 287, in wrapper\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/openai/resources/chat/completions/completions.py\", line 925, in create\n    return self._post(\n           ^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/openai/_base_client.py\", line 1239, in post\n    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))\n                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/openai/_base_client.py\", line 1034, in request\n    raise self._make_status_error_from_response(err.response) from None\nopenai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: invalid_***_123. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}\n",
#                 "exception.escaped": "False"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "generic_openai_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xb5892f649dbc612a67ccc496b85d9a6a",
#         "span_id": "0xe59fb561d29eafcf",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-03T00:01:56.506374Z",
#     "end_time": "2025-07-03T00:01:56.881642Z",
#     "status": {
#         "status_code": "ERROR",
#         "description": "AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: invalid_***_123. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample.py:109",
#         "span.type": "workflow",
#         "entity.1.name": "generic_openai_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [
#         {
#             "name": "exception",
#             "timestamp": "2025-07-03T00:01:56.881627Z",
#             "attributes": {
#                 "exception.type": "openai.AuthenticationError",
#                 "exception.message": "Error code: 401 - {'error': {'message': 'Incorrect API key provided: invalid_***_123. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}",
#                 "exception.stacktrace": "Traceback (most recent call last):\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/opentelemetry/trace/__init__.py\", line 587, in use_span\n    yield span\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/opentelemetry/sdk/trace/__init__.py\", line 1105, in start_as_current_span\n    yield span\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/src/monocle_apptrace/instrumentation/common/wrapper.py\", line 70, in monocle_wrapper_span_processor\n    return_value, span_status = monocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, False, args, kwargs)\n                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/src/monocle_apptrace/instrumentation/common/wrapper.py\", line 78, in monocle_wrapper_span_processor\n    return_value = wrapped(*args, **kwargs)\n                   ^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/openai/_utils/_utils.py\", line 287, in wrapper\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/openai/resources/chat/completions/completions.py\", line 925, in create\n    return self._post(\n           ^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/openai/_base_client.py\", line 1239, in post\n    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))\n                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/openai/_base_client.py\", line 1034, in request\n    raise self._make_status_error_from_response(err.response) from None\nopenai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: invalid_***_123. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}\n",
#                 "exception.escaped": "False"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "generic_openai_1"
#         },
#         "schema_url": ""
#     }
# }