import asyncio
import os
import time

import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from common.custom_exporter import CustomConsoleSpanExporter
from openai import OpenAI, AsyncOpenAI
from openai import AzureOpenAI

from tests.common.helpers import find_span_by_type, find_spans_by_type, validate_inference_span_events, verify_inference_span

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="generic_openai_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )

@pytest.fixture(autouse=True)
def clear_spans():
    """Clear spans before each test"""
    custom_exporter.reset()
    yield

@pytest.mark.integration()
@pytest.mark.asyncio
async def test_openai_response_api_sample_async(setup):
    openai_client = AsyncOpenAI()
    response = await openai_client.responses.create(
        model="gpt-4o-mini",
        instructions="Answer in 3 words without using special characters.",
        input=[
            {"role": "system", "content": "You are a coding assistant that talks like shakespeare."},
            {"role": "user", "content": "How do I check if a Python object is an instance of a class?"}
        ],
    )
    await asyncio.sleep(5)
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
            r"^\{\"system\": \".+\"\}$",  # Pattern for system instructions
            r"^\{\"system\": \".+\"\}$",  # Pattern for system instructions
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

@pytest.mark.integration()
@pytest.mark.asyncio
async def test_openai_response_api_sample_async_stream(setup):
    openai_client = AsyncOpenAI()
    response  = await openai_client.responses.create(
        model="gpt-4o-mini",
        instructions="Answer in 6 words without using special characters.",
        input=[
            {"role": "system", "content": "You are a coding assistant that talks like shakespeare."},
            {"role": "user", "content": "How do I check if a Python object is an instance of a class?"}
        ],
        stream=True
    )
    async for chunk in response:
        pass
    await asyncio.sleep(5)
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
            r"^\{\"system\": \".+\"\}$",  # Pattern for system instructions
            r"^\{\"system\": \".+\"\}$",  # Pattern for system instructions
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


@pytest.mark.integration()
def test_openai_response_api_sample(setup):
    openai_client = OpenAI()
    response  = openai_client.responses.create(
        model="gpt-4o-mini",
        instructions="Answer in 5 words without using special characters.",
        input=[
            {"role": "system", "content": "You are a coding assistant that talks like shakespeare."},
            {"role": "user", "content": "How do I check if a Python object is an instance of a class?"}
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
            r"^\{\"system\": \".+\"\}$",  # Pattern for system instructions
            r"^\{\"system\": \".+\"\}$",  # Pattern for system instructions
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

@pytest.mark.integration()
def test_azure_openai_response_api_sample(setup):
    azure_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-03-01-preview"
    )
    response = azure_client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a coding assistant that talks like a pirate. Answer is 10 words without using special characters",
        input="How do I check if a Python object is an instance of a class?",
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
            entity_type="inference.azure_openai",
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
            r"^\{\"system\": \".+\"\}$",  # Pattern for system instructions
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

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "openai.resources.responses.responses.AsyncResponses",
#     "context": {
#         "trace_id": "0x371b89662d25f36f28e6fcb6e6241bac",
#         "span_id": "0x2bd0dced357096ad",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc28a69e3b63cbf22",
#     "start_time": "2025-07-02T23:41:20.440275Z",
#     "end_time": "2025-07-02T23:41:21.637590Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_response_sample.py:34",
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
#             "timestamp": "2025-07-02T23:41:21.637371Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"Answer in 3 words.\"}",
#                     "{\"system\": \"You are a coding assistant that talks like shakespeare.\"}",
#                     "{\"user\": \"How do I check if a Python object is an instance of a class?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T23:41:21.637482Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"Utilize `isinstance` function.\"}",
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T23:41:21.637538Z",
#             "attributes": {
#                 "completion_tokens": 9,
#                 "prompt_tokens": 47,
#                 "total_tokens": 56
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
#         "trace_id": "0x371b89662d25f36f28e6fcb6e6241bac",
#         "span_id": "0xc28a69e3b63cbf22",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T23:41:20.440122Z",
#     "end_time": "2025-07-02T23:41:21.637642Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_response_sample.py:34",
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
#         "trace_id": "0x0773405faffaf1d4cd1da3397292a9b1",
#         "span_id": "0x0dc89a07fb4f7bf8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T23:41:26.670844Z",
#     "end_time": "2025-07-02T23:41:27.102106Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_response_sample.py:101",
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
#     "name": "openai.resources.responses.responses.AsyncResponses",
#     "context": {
#         "trace_id": "0x0773405faffaf1d4cd1da3397292a9b1",
#         "span_id": "0x6017355087644c0a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0dc89a07fb4f7bf8",
#     "start_time": "2025-07-02T23:41:26.671045Z",
#     "end_time": "2025-07-02T23:42:19.258844Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_response_sample.py:101",
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
#             "timestamp": "2025-07-02T23:41:27.101936Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"Answer in 6 words.\"}",
#                     "{\"system\": \"You are a coding assistant that talks like shakespeare.\"}",
#                     "{\"user\": \"How do I check if a Python object is an instance of a class?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T23:41:32.809336Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"Use `isinstance(object, ClassName)` fair.\"}",
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T23:42:15.557192Z",
#             "attributes": {
#                 "completion_tokens": 12,
#                 "prompt_tokens": 47,
#                 "total_tokens": 59
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
#     "name": "openai.resources.responses.responses.Responses",
#     "context": {
#         "trace_id": "0xf5f081358f96eeae8c246be373be9771",
#         "span_id": "0x117c093ccceae4cc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x1cf01f4cb6d7a17f",
#     "start_time": "2025-07-02T23:42:24.285300Z",
#     "end_time": "2025-07-02T23:42:25.060627Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_response_sample.py:171",
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
#             "timestamp": "2025-07-02T23:42:25.060438Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"Answer in 5 words.\"}",
#                     "{\"system\": \"You are a coding assistant that talks like shakespeare.\"}",
#                     "{\"user\": \"How do I check if a Python object is an instance of a class?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T23:42:25.060543Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"Use isinstance(object, ClassName).\"}",
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T23:42:25.060587Z",
#             "attributes": {
#                 "completion_tokens": 10,
#                 "prompt_tokens": 47,
#                 "total_tokens": 57
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
#         "trace_id": "0xf5f081358f96eeae8c246be373be9771",
#         "span_id": "0x1cf01f4cb6d7a17f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T23:42:24.285067Z",
#     "end_time": "2025-07-02T23:42:25.060672Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_response_sample.py:171",
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
#     "name": "openai.resources.responses.responses.Responses",
#     "context": {
#         "trace_id": "0xe805068d3e906f2369d8d59718ba2a9d",
#         "span_id": "0x958d0b7d59de6d1a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3da658a1f8bba435",
#     "start_time": "2025-07-02T23:42:30.090634Z",
#     "end_time": "2025-07-02T23:42:32.025415Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_response_sample.py:241",
#         "workflow.name": "generic_openai_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.azure_openai",
#         "entity.1.provider_name": "okahu-openai-dev.openai.azure.com",
#         "entity.1.inference_endpoint": "https://okahu-openai-dev.openai.azure.com/openai/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-02T23:42:32.025084Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a coding assistant that talks like a pirate. Answer is 10 words\"}",
#                     "{\"user\": \"How do I check if a Python object is an instance of a class?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T23:42:32.025238Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"Use 'isinstance(your_object, YourClass)' to check, matey!\"}",
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T23:42:32.025326Z",
#             "attributes": {
#                 "completion_tokens": 18,
#                 "prompt_tokens": 42,
#                 "total_tokens": 60
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
#         "trace_id": "0xe805068d3e906f2369d8d59718ba2a9d",
#         "span_id": "0x3da658a1f8bba435",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T23:42:30.090379Z",
#     "end_time": "2025-07-02T23:42:32.025515Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_openai_response_sample.py:241",
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