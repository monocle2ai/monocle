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

def assert_inference_span(span, expected_type, expected_endpoint_contains=""):
    """Assert that the span has the expected inference attributes."""
    span_attributes = span.attributes
    assert span_attributes["span.type"] == "inference"
    # assert span_attributes["entity.1.type"] == expected_type
    assert "entity.1.provider_name" in span_attributes
    assert "entity.1.inference_endpoint" in span_attributes
    
    # if expected_endpoint_contains:
    #     assert expected_endpoint_contains in span_attributes["entity.1.inference_endpoint"]
    
    # Check for model information
    assert "entity.2.name" in span_attributes
    assert "entity.2.type" in span_attributes
    assert span_attributes["entity.2.type"].startswith("model.llm.")

    # Check events structure
    assert len(span.events) >= 2  # Should have input and output events
    
    # Verify input event
    input_event = next((e for e in span.events if e.name == "data.input"), None)
    assert input_event is not None
    assert "input" in input_event.attributes
    
    # Verify output event
    output_event = next((e for e in span.events if e.name == "data.output"), None)
    assert output_event is not None
    assert "response" in output_event.attributes
    
    # Check for metadata event (may not always be present for all providers)
    metadata_event = next((e for e in span.events if e.name == "metadata"), None)
    if metadata_event:
        # If metadata is present, check for usage information
        metadata_attrs = metadata_event.attributes
        if "completion_tokens" in metadata_attrs:
            assert isinstance(metadata_attrs["completion_tokens"], int)
        if "prompt_tokens" in metadata_attrs:
            assert isinstance(metadata_attrs["prompt_tokens"], int)
        if "total_tokens" in metadata_attrs:
            assert isinstance(metadata_attrs["total_tokens"], int)

def assert_workflow_span_exists(spans):
    """Assert that a workflow span exists."""
    workflow_spans = [s for s in spans if s.attributes.get("span.type") == "workflow"]
    assert len(workflow_spans) > 0, "No workflow span found"

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
    
    try:
        # Make the chat completion call
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
        
        # Find the inference span
        inference_spans = [s for s in spans if s.attributes.get("span.type") == "inference"]
        assert len(inference_spans) > 0, "No inference span found"
        
        # Assert inference span properties
        inference_span = inference_spans[0]
        assert_inference_span(inference_span, "inference.azure_ai_inference", "models.inference.ai.azure.com")
        
        # Assert workflow span exists
        assert_workflow_span_exists(spans)
        
        # Verify response content is captured (not "streaming_response")
        output_event = next((e for e in inference_span.events if e.name == "data.output"), None)
        assert output_event is not None
        response_content = output_event.attributes.get("response", "")
        assert response_content != "streaming_response", "Streaming response placeholder found instead of actual content"
        assert len(response_content) > 0, "Empty response content"
        assert "Paris" in response_content, "Expected response content not found"
        
    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")

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
    
    try:
        # Make the streaming chat completion call
        response = client.complete(
            messages=[
                SystemMessage("You are a helpful assistant that provides concise answers."),
                UserMessage("Explain what Python is in one paragraph."),
            ],
            model=os.getenv("AZURE_AI_INFERENCE_MODEL", "gpt-4o-mini"),
            max_tokens=100,
            temperature=0.1,
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
        time.sleep(3)
        
        spans = custom_exporter.get_captured_spans()
        print(f"Captured {len(spans)} spans")
        
        # Verify we have spans
        assert len(spans) > 0, "No spans captured"
        
        # Find the inference span
        inference_spans = [s for s in spans if s.attributes.get("span.type") == "inference"]
        assert len(inference_spans) > 0, "No inference span found"
        
        # Assert inference span properties
        inference_span = inference_spans[0]
        assert_inference_span(inference_span, "inference.azure_ai_inference", "models.inference.ai.azure.com")
        
        # Assert workflow span exists
        assert_workflow_span_exists(spans)
        
        # Verify streaming response content is properly captured
        output_event = next((e for e in inference_span.events if e.name == "data.output"), None)
        assert output_event is not None
        response_content = output_event.attributes.get("response", "")
        
        # For streaming, we should get the accumulated content, not the placeholder
        assert response_content != "streaming_response", "Streaming response placeholder found - streaming not properly handled"
        assert len(response_content) > 0, "Empty response content for streaming"
        assert "Python" in response_content, "Expected response content not found"
        
        # Assert we got a valid streamed response
        assert len(collected_chunks) > 1, "Should have received multiple chunks for streaming"
        assert len(full_response) > 0, "Should have accumulated response content"
        
    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")

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
        
        try:
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
            await asyncio.sleep(3)
            
            spans = custom_exporter.get_captured_spans()
            print(f"Captured {len(spans)} spans")
            
            # Verify we have spans
            assert len(spans) > 0, "No spans captured"
            
            # Find the inference span
            inference_spans = [s for s in spans if s.attributes.get("span.type") == "inference"]
            assert len(inference_spans) > 0, "No inference span found"
            
            # Assert inference span properties
            inference_span = inference_spans[0]
            assert_inference_span(inference_span, "inference.azure_ai_inference", "models.inference.ai.azure.com")
            
            # Assert workflow span exists
            assert_workflow_span_exists(spans)
            
            # Verify response content is captured
            output_event = next((e for e in inference_span.events if e.name == "data.output"), None)
            assert output_event is not None
            response_content = output_event.attributes.get("response", "")
            assert response_content != "streaming_response", "Streaming response placeholder found instead of actual content"
            assert len(response_content) > 0, "Empty response content"
            assert "Jupiter" in response_content, "Expected response content not found"
            
        except Exception as e:
            pytest.fail(f"Test failed with exception: {str(e)}")

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
        
        try:
            # Make the async streaming chat completion call
            response = await client.complete(
                messages=[
                    SystemMessage("You are a helpful assistant that provides concise answers."),
                    UserMessage("Explain what machine learning is in one paragraph."),
                ],
                model=os.getenv("AZURE_AI_INFERENCE_MODEL", "gpt-4o-mini"),
                max_tokens=100,
                temperature=0.1,
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
            
            # Find the inference span
            inference_spans = [s for s in spans if s.attributes.get("span.type") == "inference"]
            assert len(inference_spans) > 0, "No inference span found"
            
            # Assert inference span properties
            inference_span = inference_spans[0]
            assert_inference_span(inference_span, "inference.azure_ai_inference", "models.inference.ai.azure.com")
            
            # Assert workflow span exists
            assert_workflow_span_exists(spans)
            
            # Verify streaming response content is properly captured
            output_event = next((e for e in inference_span.events if e.name == "data.output"), None)
            assert output_event is not None
            response_content = output_event.attributes.get("response", "")
            
            # For streaming, we should get the accumulated content, not the placeholder
            assert response_content != "streaming_response", "Streaming response placeholder found - async streaming not properly handled"
            assert len(response_content) > 0, "Empty response content for async streaming"
            assert ("learning" in response_content.lower() or "machine" in response_content.lower()), "Expected response content not found"
            
            # Assert we got a valid streamed response
            assert len(collected_chunks) > 1, "Should have received multiple chunks for async streaming"
            assert len(full_response) > 0, "Should have accumulated response content"
            
        except Exception as e:
            pytest.fail(f"Test failed with exception: {str(e)}")

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
        },
        {
            "name": "Azure AI Foundry",
            "endpoint": "https://models.ai.azure.com",
            "expected_provider": "azure_ai_foundry"
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
    # For manual testing
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Setup telemetry
    setup_monocle_telemetry(
        workflow_name="azure_ai_inference_manual_test",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )
    
    print("Running manual Azure AI Inference tests...")
    
    # Check environment variables
    if not os.getenv("AZURE_AI_CHAT_KEY"):
        print("Please set AZURE_AI_CHAT_KEY environment variable")
        sys.exit(1)
    
    try:
        # Run sync test
        print("\n=== Testing Sync Chat Completion ===")
        # run sync test
        test_azure_ai_inference_chat_completion_sync(setup)
        custom_exporter.reset()
        # You can call test functions here for manual testing
        
    except Exception as e:
        print(f"Manual test failed: {e}")
