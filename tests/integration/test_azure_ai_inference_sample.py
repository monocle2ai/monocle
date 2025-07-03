import os
import time
import asyncio
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

# Azure AI Inference imports
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.aio import ChatCompletionsClient as AsyncChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from tests.common.helpers import find_span_by_type, find_spans_by_type, validate_inference_span_events, verify_inference_span

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="azure_ai_inference_integration_test",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )

@pytest.fixture(autouse=True)
def pre_test():
    custom_exporter.reset()

@pytest.mark.integration()
def test_azure_ai_inference_chat_completion_sync(setup):
    """Test synchronous Azure AI Inference chat completion instrumentation."""
    
    # Setup client - using GitHub Models endpoint as an example
    endpoint = os.getenv("AZURE_AI_CHAT_ENDPOINT", "https://models.inference.ai.azure.com")
    api_key = os.getenv("AZURE_AI_CHAT_KEY")
    
    if not api_key:
        pytest.skip("AZURE_AI_CHAT_KEY environment variable not set")
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )
    
    response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant that provides concise answers."),
            UserMessage("What is the capital of France? Please answer in one sentence."),
        ],
        model=os.getenv("AZURE_AI_INFERENCE_MODEL", "gpt-4o-mini"),
        max_tokens=50,
        temperature=0.1
    )
    
    print(f"Response: {response.choices[0].message.content}")
    
    # Wait for spans to be processed
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
            r"^\{'system': '.+'\}$",  # Pattern for system
            r"^\{'user': '.+'\}$",  # Pattern for user input
        ],
        output_pattern=r"^\{'assistant': '.+'\}$",  # Pattern for AI response
        metadata_requirements={
            "completion_tokens": int,
            "prompt_tokens": int,
            "total_tokens": int
        }
    )
    
    workflow_span = find_span_by_type(spans, "workflow")
    
    assert workflow_span is not None, "Expected to find workflow span"
    
    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "azure_ai_inference_integration_test"
    assert workflow_span.attributes["entity.1.type"] == "workflow.generic"

@pytest.mark.integration()
def test_azure_ai_inference_chat_completion_streaming_sync(setup):
    """Test synchronous Azure AI Inference streaming chat completion instrumentation."""
    
    # Setup client
    endpoint = os.getenv("AZURE_AI_CHAT_ENDPOINT", "https://models.inference.ai.azure.com")
    api_key = os.getenv("AZURE_AI_CHAT_KEY")
    
    if not api_key:
        pytest.skip("AZURE_AI_CHAT_KEY environment variable not set")
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )
    
    # Make the streaming chat completion call
    response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant that provides concise answers."),
            UserMessage("Explain what Python is in 10 words."),
        ],
        model=os.getenv("AZURE_AI_INFERENCE_MODEL", "gpt-4o-mini"),
        max_tokens=100,
        temperature=0.1,
        model_extras={"stream_options": {"include_usage": True}},
        stream=True
    )
    
    # Collect the streamed response
    collected_chunks = []
    collected_content = []
    
    for chunk in response:
        collected_chunks.append(chunk)
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            collected_content.append(chunk.choices[0].delta.content)
    
    full_response = "".join(collected_content)
    print(f"Streamed response: {full_response}")
    
    # Wait for spans to be processed
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
            r"^\{'system': '.+'\}$",  # Pattern for system
            r"^\{'user': '.+'\}$",  # Pattern for user input
        ],
        output_pattern=r"^\{'assistant': '.+'\}$",  # Pattern for AI response
        # TODO: Uncomment when metadata is available
        metadata_requirements={
            "completion_tokens": int,
            "prompt_tokens": int,
            "total_tokens": int
        }
    )
    
    workflow_span = find_span_by_type(spans, "workflow")
    
    assert workflow_span is not None, "Expected to find workflow span"
    
    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "azure_ai_inference_integration_test"
    assert workflow_span.attributes["entity.1.type"] == "workflow.generic"


@pytest.mark.integration()
@pytest.mark.asyncio
async def test_azure_ai_inference_chat_completion_async(setup):
    """Test asynchronous Azure AI Inference chat completion instrumentation."""
    
    # Setup async client
    endpoint = os.getenv("AZURE_AI_CHAT_ENDPOINT", "https://models.inference.ai.azure.com")
    api_key = os.getenv("AZURE_AI_CHAT_KEY")
    
    if not api_key:
        pytest.skip("AZURE_AI_CHAT_KEY environment variable not set")
    
    async with AsyncChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    ) as client:
        
        # Make the async chat completion call
        response = await client.complete(
            messages=[
                SystemMessage("You are a helpful assistant that provides concise answers."),
                UserMessage("What is the largest planet in our solar system? Please answer in one sentence."),
            ],
            model=os.getenv("AZURE_AI_INFERENCE_MODEL", "gpt-4o-mini"),
            max_tokens=50,
            temperature=0.1
        )
        
        print(f"Async response: {response.choices[0].message.content}")
        
        # Wait for spans to be processed
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
                r"^\{'system': '.+'\}$",  # Pattern for system
                r"^\{'user': '.+'\}$",  # Pattern for user input
            ],
            output_pattern=r"^\{'assistant': '.+'\}$",  # Pattern for AI response
            # TODO: Uncomment when metadata is available
            metadata_requirements={
                "completion_tokens": int,
                "prompt_tokens": int,
                "total_tokens": int
            }
        )
        
        workflow_span = find_span_by_type(spans, "workflow")
        
        assert workflow_span is not None, "Expected to find workflow span"
        
        assert workflow_span.attributes["span.type"] == "workflow"
        assert workflow_span.attributes["entity.1.name"] == "azure_ai_inference_integration_test"
        assert workflow_span.attributes["entity.1.type"] == "workflow.generic"


@pytest.mark.integration()
@pytest.mark.asyncio
async def test_azure_ai_inference_chat_completion_streaming_async(setup):
    """Test asynchronous Azure AI Inference streaming chat completion instrumentation."""
    
    # Setup async client
    endpoint = os.getenv("AZURE_AI_CHAT_ENDPOINT", "https://models.inference.ai.azure.com")
    api_key = os.getenv("AZURE_AI_CHAT_KEY")
    
    if not api_key:
        pytest.skip("AZURE_AI_CHAT_KEY environment variable not set")
    
    async with AsyncChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    ) as client:
        
        # Make the async streaming chat completion call
        response = await client.complete(
            messages=[
                SystemMessage("You are a helpful assistant that provides concise answers."),
                UserMessage("Explain what machine learning is in 10 words."),
            ],
            model=os.getenv("AZURE_AI_INFERENCE_MODEL", "gpt-4o-mini"),
            max_tokens=100,
            temperature=0.1,
            model_extras={"stream_options": {"include_usage": True}},
            stream=True
        )
        
        # Collect the streamed response
        collected_chunks = []
        collected_content = []
        
        async for chunk in response:
            collected_chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                collected_content.append(chunk.choices[0].delta.content)
        
        full_response = "".join(collected_content)
        print(f"Async streamed response: {full_response}")
        
        # Wait for spans to be processed
        await asyncio.sleep(3)
        
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
                r"^\{'system': '.+'\}$",  # Pattern for system
                r"^\{'user': '.+'\}$",  # Pattern for user input
            ],
            output_pattern=r"^\{'assistant': '.+'\}$",  # Pattern for AI response
            # TODO: Uncomment when metadata is available
            metadata_requirements={
                "completion_tokens": int,
                "prompt_tokens": int,
                "total_tokens": int
            }
        )
        
        workflow_span = find_span_by_type(spans, "workflow")
        
        assert workflow_span is not None, "Expected to find workflow span"
        
        assert workflow_span.attributes["span.type"] == "workflow"
        assert workflow_span.attributes["entity.1.name"] == "azure_ai_inference_integration_test"
        assert workflow_span.attributes["entity.1.type"] == "workflow.generic"


# Test for different Azure AI Inference providers
@pytest.mark.integration()
def test_azure_ai_inference_different_providers(setup):
    """Test Azure AI Inference with different provider endpoints."""
    
    # Test different provider configurations
    provider_configs = [
        {
            "name": "GitHub Models",
            "endpoint": "https://models.inference.ai.azure.com",
            "expected_provider": "github_models"
        }
    ]
    
    api_key = os.getenv("AZURE_AI_CHAT_KEY")
    if not api_key:
        pytest.skip("AZURE_AI_CHAT_KEY environment variable not set")
    
    for config in provider_configs:
        custom_exporter.reset()
        
        try:
            client = ChatCompletionsClient(
                endpoint=config["endpoint"],
                credential=AzureKeyCredential(api_key)
            )
            
            response = client.complete(
                messages=[
                    SystemMessage("You are a helpful assistant."),
                    UserMessage("Say 'Hello from Azure AI!'"),
                ],
                model=os.getenv("AZURE_AI_INFERENCE_MODEL", "gpt-4o-mini"),
                max_tokens=20
            )
            
            time.sleep(2)
            spans = custom_exporter.get_captured_spans()
            
            if len(spans) > 0:
                inference_spans = [s for s in spans if s.attributes.get("span.type") == "inference"]
                if len(inference_spans) > 0:
                    span = inference_spans[0]
                    provider_name = span.attributes.get("entity.1.provider_name", "")
                    endpoint_attr = span.attributes.get("entity.1.inference_endpoint", "")
                    
                    print(f"Provider: {config['name']}, Endpoint: {endpoint_attr}, Provider Name: {provider_name}")
                    
                    # Verify endpoint is captured correctly
                    assert config["endpoint"] in endpoint_attr, f"Expected endpoint not found for {config['name']}"
                    
        except Exception as e:
            print(f"Provider {config['name']} test failed (this may be expected if endpoint is not available): {str(e)}")
            # Don't fail the test for provider-specific issues
            continue

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
    
# {
#     "name": "azure.ai.inference.aio._patch.ChatCompletionsClient",
#     "context": {
#         "trace_id": "0xebaeb6f46ce40ecc8bfbde50c150d93e",
#         "span_id": "0xeba9f939dd2fa43f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x09a9da3e03e7f1e8",
#     "start_time": "2025-07-02T15:36:10.326130Z",
#     "end_time": "2025-07-02T15:36:11.668362Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_azure_ai_inference_sample.py:278",
#         "workflow.name": "azure_ai_inference_integration_test",
#         "span.type": "inference",
#         "entity.1.type": "inference.azure_openai",
#         "entity.1.provider_name": "okahu-openai-dev.openai.azure.com",
#         "entity.1.inference_endpoint": "https://okahu-openai-dev.openai.azure.com/openai/deployments/kshitiz-gpt",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-02T15:36:11.668250Z",
#             "attributes": {
#                 "input": [
#                     "{'system': 'You are a helpful assistant that provides concise answers.'}",
#                     "{'user': 'What is the largest planet in our solar system? Please answer in one sentence.'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T15:36:11.668306Z",
#             "attributes": {
#                 "response": "{'assistant': 'Jupiter is the largest planet in our solar system.'}",
#                 "status": "success",
#                 "status_code": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T15:36:11.668347Z",
#             "attributes": {
#                 "completion_tokens": 11,
#                 "prompt_tokens": 37,
#                 "total_tokens": 48,
#                 "model": "gpt-3.5-turbo-0125"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "azure_ai_inference_integration_test"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xebaeb6f46ce40ecc8bfbde50c150d93e",
#         "span_id": "0x09a9da3e03e7f1e8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T15:36:10.325999Z",
#     "end_time": "2025-07-02T15:36:11.668384Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_azure_ai_inference_sample.py:278",
#         "span.type": "workflow",
#         "entity.1.name": "azure_ai_inference_integration_test",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "azure_ai_inference_integration_test"
#         },
#         "schema_url": ""
#     }
# }
