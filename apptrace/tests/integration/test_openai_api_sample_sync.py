import logging
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from openai import OpenAI
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)
@pytest.fixture(scope="module")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="generic_openai_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_openai_api_sample(setup):
    openai = OpenAI()
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions in 10 words or less",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )
    time.sleep(5)
    
    spans = setup.get_captured_spans()
    logger.info(f"Captured {len(spans)} spans")
    
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
    assert workflow_span.attributes["entity.1.type"] == "workflow.openai"


def test_openai_api_sample_stream(setup):
    openai = OpenAI()
    stream = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions in 10 words or less",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
        stream=True,
        stream_options= {"include_usage": True},
    )

    # Collect the streamed response
    collected_chunks = []
    collected_messages = []

    for chunk in stream:
        collected_chunks.append(chunk)
        if chunk.choices and chunk.choices[0].delta.content is not None:
            collected_messages.append(chunk.choices[0].delta.content)

    full_response = "".join(collected_messages)

    # Wait for spans to be processed
    time.sleep(5)

    spans = setup.get_captured_spans()
    logger.info(f"Captured {len(spans)} spans")
    
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
    assert workflow_span.attributes["entity.1.type"] == "workflow.openai"
    
# run something after each test
@pytest.fixture(autouse=True)
def teardown_function(setup):
    """Teardown function to clear the exporter after each test."""
    setup.reset()

if __name__ == "__main__":
    # Run the tests using pytest
    pytest.main([__file__, "-s", "--tb=short"])
    
# {
#     "name": "openai.resources.chat.completions.completions.Completions",
#     "context": {
#         "trace_id": "0x11ed6c1e65a5f8e8280e44de2f3500c5",
#         "span_id": "0x7004a75f87fd8a5a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x144e04a20fa7ea69",
#     "start_time": "2025-07-02T21:07:27.856755Z",
#     "end_time": "2025-07-02T21:07:30.197008Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample_sync.py:27",
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
#             "timestamp": "2025-07-02T21:07:30.196826Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a helpful assistant to answer coffee related questions\"}",
#                     "{\"user\": \"What is an americano?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T21:07:30.196905Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"An Americano is a type of coffee drink made by diluting a shot (or shots) of espresso. \"}",
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T21:07:30.196935Z",
#             "attributes": {
#                 "completion_tokens": 96,
#                 "prompt_tokens": 26,
#                 "total_tokens": 122
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
#         "trace_id": "0x11ed6c1e65a5f8e8280e44de2f3500c5",
#         "span_id": "0x144e04a20fa7ea69",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T21:07:27.856700Z",
#     "end_time": "2025-07-02T21:07:30.197079Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample_sync.py:27",
#         "span.type": "workflow",
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
# .{
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x0a721fae8fb17dcdedeca45bb51280af",
#         "span_id": "0x3ff825c0640997fb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T21:07:35.220751Z",
#     "end_time": "2025-07-02T21:07:36.983826Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample_sync.py:96",
#         "span.type": "workflow",
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
#         "trace_id": "0x0a721fae8fb17dcdedeca45bb51280af",
#         "span_id": "0x8409d7fc07b2706d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3ff825c0640997fb",
#     "start_time": "2025-07-02T21:07:35.220865Z",
#     "end_time": "2025-07-02T21:07:37.982196Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample_sync.py:96",
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
#             "timestamp": "2025-07-02T21:07:36.983770Z",
#             "attributes": {
#                 "input": [
#                     "{'system': 'You are a helpful assistant to answer coffee related questions'}",
#                     "{'user': 'What is an americano?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T21:07:36.984621Z",
#             "attributes": {
#                 "response": "{'assistant': 'An Americano is a type of coffee drink made by diluting espresso with hot water. The process typically involves brewing one or two shots of espresso and then adding hot water to achieve the desired strength and flavor. The result is a coffee that has a similar strength to drip coffee but retains the rich flavor profile of espresso. The Americano is often enjoyed black, but it can also be customized with milk, cream, or sweeteners according to personal preference.'}",
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T21:07:37.981816Z",
#             "attributes": {
#                 "completion_tokens": 91,
#                 "prompt_tokens": 26,
#                 "total_tokens": 117
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
