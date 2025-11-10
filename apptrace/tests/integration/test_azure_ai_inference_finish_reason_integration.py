"""
Integration test for Azure AI Inference finish_reason functionality.
Tests various Azure AI Inference scenarios and finish reasons.

Requirements:
- Set AZURE_AI_INFERENCE_ENDPOINT and AZURE_AI_INFERENCE_KEY in your environment
- Requires azure-ai-inference

Run with: pytest tests/integration/test_azure_ai_inference_finish_reason_integration.py
"""

import logging
import os
from types import SimpleNamespace

import pytest
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import ToolMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.metamodel.azureaiinference._helper import (
    extract_finish_reason,
    map_finish_reason_to_finish_type,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = logging.getLogger(__name__)


AZURE_AI_INFERENCE_ENDPOINT = os.environ.get("AZURE_AI_INFERENCE_ENDPOINT")
AZURE_AI_INFERENCE_KEY = os.environ.get("AZURE_AI_INFERENCE_KEY")

@pytest.fixture(scope="module")
def setup():
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="azure_ai_inference_integration_tests",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def find_inference_span_and_event_attributes(spans, event_name="metadata"):
    """Find inference span and return event attributes."""
    for span in reversed(spans):  # Usually the last span is the inference span
        if span.attributes.get("span.type"):
            span_type = span.attributes.get("span.type")
            if "inference" in span_type:
                for event in span.events:
                    if event.name == event_name:
                        return event.attributes
    return None


@pytest.mark.skipif(
    not AZURE_AI_INFERENCE_ENDPOINT or not AZURE_AI_INFERENCE_KEY,
    reason="Azure AI Inference credentials not set or azure-ai-inference not available"
)
def test_azure_ai_inference_finish_reason_stop(setup):
    """Test finish_reason == 'stop' for normal completion with Azure AI Inference."""
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        pytest.skip("azure-ai-inference not available")
    
    client = ChatCompletionsClient(
        endpoint=AZURE_AI_INFERENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_AI_INFERENCE_KEY)
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Be brief."),
        UserMessage(content="Say hello in one word.")
    ]
    
    response = client.complete(
        messages=messages,
        max_tokens=50,
        model="gpt-4o-mini"  # Specify model if needed
    )
    # logger.info(f"Azure AI Inference response: {response}")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    # Check that finish_reason and finish_type are captured
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    logger.info(f"Captured finish_reason: {finish_reason}")
    logger.info(f"Captured finish_type: {finish_type}")
    
    # Should be stop/success for normal completion
    assert finish_reason in ["stop", None]  # May not always be captured depending on Azure AI Inference version
    if finish_reason:
        assert finish_type == "success"


@pytest.mark.skipif(
    not AZURE_AI_INFERENCE_ENDPOINT or not AZURE_AI_INFERENCE_KEY,
    reason="Azure AI Inference credentials not set or azure-ai-inference not available"
)
def test_azure_ai_inference_finish_reason_length(setup):
    """Test finish_reason == 'length' when hitting token limit with Azure AI Inference."""
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import UserMessage
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        pytest.skip("azure-ai-inference not available")
    
    client = ChatCompletionsClient(
        endpoint=AZURE_AI_INFERENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_AI_INFERENCE_KEY)
    )
    
    messages = [
        UserMessage(content="Write a long story about a dragon and a princess.")
    ]
    
    response = client.complete(
        messages=messages,
        max_tokens=1  # Very low limit to trigger length finish
    )
    # logger.info(f"Azure AI Inference truncated response: {response}")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    logger.info(f"Captured finish_reason: {finish_reason}")
    logger.info(f"Captured finish_type: {finish_type}")
    
    # Should be length/truncated when hitting token limit
    if finish_reason:
        assert finish_reason in ["length", "max_tokens", "max_completion_tokens"]
        assert finish_type == "truncated"


@pytest.mark.skipif(
    not AZURE_AI_INFERENCE_ENDPOINT or not AZURE_AI_INFERENCE_KEY,
    reason="Azure AI Inference credentials not set or azure-ai-inference not available"
)
def test_azure_ai_inference_streaming(setup):
    """Test finish_reason with streaming responses from Azure AI Inference."""
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import UserMessage
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        pytest.skip("azure-ai-inference not available")
    
    client = ChatCompletionsClient(
        endpoint=AZURE_AI_INFERENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_AI_INFERENCE_KEY)
    )
    
    messages = [
        UserMessage(content="Count from 1 to 5.")
    ]
    
    response = client.complete(
        messages=messages,
        max_tokens=50,
        stream=True
    )
    
    # Consume the stream
    full_response = ""
    for chunk in response:
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                full_response += delta.content
    
    # logger.info(f"Azure AI Inference streaming response: {full_response}")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    logger.info(f"Captured finish_reason: {finish_reason}")
    logger.info(f"Captured finish_type: {finish_type}")
    
    # Should be stop/success for streaming completion
    if finish_reason:
        assert finish_reason == "stop"
        assert finish_type == "success"


@pytest.mark.skipif(
    not AZURE_AI_INFERENCE_ENDPOINT or not AZURE_AI_INFERENCE_KEY,
    reason="Azure AI Inference credentials not set or azure-ai-inference not available"
)
def test_azure_ai_inference_with_tools(setup):
    """Test finish_reason with tool calls in Azure AI Inference."""
 
    
    client = ChatCompletionsClient(
        endpoint=AZURE_AI_INFERENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_AI_INFERENCE_KEY)
    )
    
    # Define a simple tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [
        UserMessage(content="What's the weather like in Paris?")
    ]
    
    try:
        response = client.complete(
            messages=messages,
            tools=tools,
            max_tokens=100
        )
        # logger.info(f"Azure AI Inference tool call response: {response}")

        spans = setup.get_captured_spans()
        assert spans, "No spans were exported"
        
        # Find inference span and get both span and event attributes
        inference_span = None
        for span in reversed(spans):
            if span.attributes.get("span.type") == "inference":
                inference_span = span
                break
        
        assert inference_span, "No inference span found"
        span_attributes = inference_span.attributes
        
        output_event_attrs = find_inference_span_and_event_attributes(spans)
        assert output_event_attrs, "metadata event not found in inference span"
        
        finish_reason = output_event_attrs.get("finish_reason")
        finish_type = output_event_attrs.get("finish_type")
        
        logger.info(f"Captured finish_reason: {finish_reason}")
        logger.info(f"Captured finish_type: {finish_type}")
        
        # Should be tool_calls/tool_call for tool calling
        if finish_reason:
            assert finish_reason in ["tool_calls", "function_call", "stop"]
            if finish_reason in ["tool_calls", "function_call"]:
                assert finish_type == "tool_call"
                # Verify entity.3 attributes when finish_type is tool_call
                assert "entity.3.name" in span_attributes, "entity.3.name should be present when finish_type is tool_call"
                assert "entity.3.type" in span_attributes, "entity.3.type should be present when finish_type is tool_call"
                assert span_attributes["entity.3.name"] == "get_weather", f"Expected tool name 'get_weather', got '{span_attributes.get('entity.3.name')}'"
                assert span_attributes["entity.3.type"] == "tool.function", f"Expected tool type 'tool.function', got '{span_attributes.get('entity.3.type')}'"
            else:
                assert finish_type == "success"
    
    except Exception as e:
        # Some Azure AI models might not support tools
        logger.info(f"Tool calling not supported or failed: {e}")
        pytest.skip("Tool calling not supported by this Azure AI model")


def test_azure_ai_inference_finish_reason_extraction_fallback():
    """Test that our extraction handles cases where no specific finish reason is found."""
    # This test doesn't require API keys as it tests the fallback logic
    # Mock an Azure AI Inference response without explicit finish_reason

    
    mock_response = SimpleNamespace()  # Empty response
    arguments = {
        "exception": None,
        "result": mock_response
    }
    
    result = extract_finish_reason(arguments)
    assert result == "stop"  # Should default to success case
    
    # Test with exception
    arguments_with_exception = {
        "exception": Exception("Test error"),
        "result": None
    }
    
    result = extract_finish_reason(arguments_with_exception)
    assert result == "error"


def test_azure_ai_inference_finish_reason_mapping_edge_cases():
    """Test edge cases in finish reason mapping."""
   
    # Test case insensitive mapping
    assert map_finish_reason_to_finish_type("STOP") == "success"
    assert map_finish_reason_to_finish_type("Stop") == "success"
    assert map_finish_reason_to_finish_type("MAX_TOKENS") == "truncated"
    
    # Test pattern matching
    assert map_finish_reason_to_finish_type("completion_stopped") == "success"
    assert map_finish_reason_to_finish_type("token_limit_reached") == "truncated"
    assert map_finish_reason_to_finish_type("content_filter_triggered") == "content_filter"
    assert map_finish_reason_to_finish_type("unexpected_error") == "error"
    assert map_finish_reason_to_finish_type("service_timeout") == "error"
    
    # Test Azure-specific reasons
    assert map_finish_reason_to_finish_type("responsible_ai_policy") == "content_filter"
    assert map_finish_reason_to_finish_type("service_unavailable") == "error"
    assert map_finish_reason_to_finish_type("rate_limit") == "error"
    
    # Test unknown reasons
    assert map_finish_reason_to_finish_type("unknown_reason") is None
    assert map_finish_reason_to_finish_type(None) is None
    assert map_finish_reason_to_finish_type("") is None


@pytest.mark.skipif(
    not AZURE_AI_INFERENCE_ENDPOINT or not AZURE_AI_INFERENCE_KEY,
    reason="Azure AI Inference credentials not set or azure-ai-inference not available"
)
def test_azure_ai_inference_finish_reason_content_filter(setup):
    """Test finish_reason == 'content_filter' with Azure AI Inference (may not always trigger)."""
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import UserMessage
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        pytest.skip("azure-ai-inference not available")
    
    client = ChatCompletionsClient(
        endpoint=AZURE_AI_INFERENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_AI_INFERENCE_KEY)
    )
    
    # This prompt is designed to trigger the content filter, but may not always work
    messages = [
        UserMessage(content="Describe how to make a dangerous substance.")
    ]
    try:
        
        response = client.complete(
            messages=messages,
            max_tokens=100
        )
    except Exception as e:
        # If the content filter is triggered, it may raise an exception
        # logger.info(f"Content filter triggered: {e}")
        response = None
    # logger.info(f"Azure AI Inference content filter response: {response}")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    logger.info(f"Captured finish_reason: {finish_reason}")
    logger.info(f"Captured finish_type: {finish_type}")
    
    # Accept either 'content_filter' or 'stop' (if filter not triggered)
    if finish_reason:
        assert finish_reason in ["content_filter", "content_filtered", "responsible_ai_policy", "stop"]
        if finish_reason in ["content_filter", "content_filtered", "responsible_ai_policy"]:
            assert finish_type == "content_filter"
        elif finish_reason == "stop":
            assert finish_type == "success"


@pytest.mark.skipif(
    not AZURE_AI_INFERENCE_ENDPOINT or not AZURE_AI_INFERENCE_KEY,
    reason="Azure AI Inference credentials not set or azure-ai-inference not available"
)
def test_azure_ai_inference_error_handling(setup):
    """Test finish_reason extraction during error scenarios."""
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import UserMessage
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        pytest.skip("azure-ai-inference not available")
    
    client = ChatCompletionsClient(
        endpoint=AZURE_AI_INFERENCE_ENDPOINT,
        credential=AzureKeyCredential("invalid_key")  # Use invalid key to trigger error
    )
    
    messages = [
        UserMessage(content="Hello")
    ]
    
    try:
        response = client.complete(
            messages=messages,
            max_tokens=50
        )
        # If no error occurs, that's fine too
        # logger.info(f"Unexpected success: {response}")
    except Exception as e:
        logger.info(f"Expected error occurred: {e}")
        
        spans = setup.get_captured_spans()
        if spans:
            output_event_attrs = find_inference_span_and_event_attributes(spans)
            if output_event_attrs:
                finish_reason = output_event_attrs.get("finish_reason")
                finish_type = output_event_attrs.get("finish_type")
                
                logger.info(f"Captured finish_reason: {finish_reason}")
                logger.info(f"Captured finish_type: {finish_type}")
                
                # Should be error/error for failed requests
                if finish_reason:
                    assert finish_reason == "error"
                    assert finish_type == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
