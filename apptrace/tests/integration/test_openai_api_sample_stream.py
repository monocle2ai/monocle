import asyncio
import logging

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from openai import AsyncOpenAI
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

@pytest.fixture(scope="function")
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


@pytest.mark.asyncio
async def test_openai_api_sample(setup):
    openai = AsyncOpenAI()
    response = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions in 10 words or less",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )
    await asyncio.sleep(5)

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


@pytest.mark.asyncio
async def test_openai_api_sample_stream(setup):
    openai = AsyncOpenAI()
    stream = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions and answer in 10 words or less",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
        stream=True,
        stream_options= {"include_usage": True},
    )

    # Collect the streamed response
    collected_chunks = []
    collected_messages = []

    async for chunk in stream:
        collected_chunks.append(chunk)
        if chunk.choices and chunk.choices[0].delta.content is not None:
            collected_messages.append(chunk.choices[0].delta.content)

    full_response = "".join(collected_messages)
    logger.info(f"Streamed response: {full_response}")

    # Wait for spans to be processed
    await asyncio.sleep(5)

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

    # Assert we got a valid response
    assert len(collected_chunks) > 0
    assert len(full_response) > 0
    


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "openai.resources.chat.completions.completions.AsyncCompletions",
#     "context": {
#         "trace_id": "0x06eecf4252ac8b67f1cb39822a5d40ca",
#         "span_id": "0x135a7782f4551c33",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xbc618fdf7a1dbde8",
#     "start_time": "2025-07-02T21:00:11.304848Z",
#     "end_time": "2025-07-02T21:00:13.922418Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample_stream.py:33",
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
#             "timestamp": "2025-07-02T21:00:13.922309Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a helpful assistant to answer coffee related questions\"}",
#                     "{\"user\": \"What is an americano?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T21:00:13.922357Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"An Americano is a type of coffee drink made by diluting espresso with hot water. \"}",
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T21:00:13.922379Z",
#             "attributes": {
#                 "completion_tokens": 125,
#                 "prompt_tokens": 26,
#                 "total_tokens": 151
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
#         "trace_id": "0x06eecf4252ac8b67f1cb39822a5d40ca",
#         "span_id": "0xbc618fdf7a1dbde8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T21:00:11.304797Z",
#     "end_time": "2025-07-02T21:00:13.922465Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample_stream.py:33",
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
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xafa38b0906415d29025076ee05e8dc96",
#         "span_id": "0x62549e039822fcfa",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T21:00:18.948688Z",
#     "end_time": "2025-07-02T21:00:20.232554Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample_stream.py:103",
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
#     "name": "openai.resources.chat.completions.completions.AsyncCompletions",
#     "context": {
#         "trace_id": "0xafa38b0906415d29025076ee05e8dc96",
#         "span_id": "0xd8b9486a219e3f36",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x62549e039822fcfa",
#     "start_time": "2025-07-02T21:00:18.948810Z",
#     "end_time": "2025-07-02T21:00:20.351935Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_api_sample_stream.py:103",
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
#             "timestamp": "2025-07-02T21:00:20.232405Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a helpful assistant to answer coffee related questions and answer in 10 words or less\"}",
#                     "{\"user\": \"What is an americano?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T21:00:20.234867Z",
#             "attributes": {
#                 "response": "{'assistant': 'Espresso diluted with hot water.'}",
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T21:00:20.347901Z",
#             "attributes": {
#                 "completion_tokens": 7,
#                 "prompt_tokens": 34,
#                 "total_tokens": 41
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