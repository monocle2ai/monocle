"""
Integration test for AWS Bedrock finish_reason (stopReason) and finish_type using the real AWS Bedrock API.
Tests: end_turn, max_tokens, content_filter, tool_use, stop_sequence, error conditions.

Requirements:
- AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
- AWS Bedrock access enabled for the models being tested
- Requires boto3>=1.28.0

Run with: pytest tests/integration/test_bedrock_finish_reason_integration.py
"""
import logging
import os

import boto3
import pytest
from botocore.exceptions import ClientError
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.metamodel.botocore.handlers.botocore_span_handler import (
    BotoCoreSpanHandler,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

logger = logging.getLogger(__name__)

# Setup telemetry
@pytest.fixture(scope="module")
def setup():
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="bedrock_integration_tests",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
            span_handlers={"botocore_handler": BotoCoreSpanHandler()},
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

# Default models to test (can be overridden via environment variables)
AI21_MODEL = os.environ.get("BEDROCK_AI21_MODEL", "ai21.jamba-1-5-mini-v1:0")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

def get_bedrock_client():
    """Get Bedrock runtime client."""
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)

def find_inference_span_and_event_attributes(spans, event_name="metadata"):
    """Find inference span and extract event attributes."""
    for span in reversed(spans):  # Usually the last span is the inference span
        if span.attributes.get("span.type") == "inference":
            for event in span.events:
                if event.name == event_name:
                    return event.attributes
    return None


def test_bedrock_finish_reason_end_turn(setup):
    """Test stopReason == 'end_turn' for a normal completion using Claude via Bedrock."""
    client = get_bedrock_client()
    
    response = client.converse(
        modelId=AI21_MODEL,
        messages=[
            {
                "role": "user",
                "content": [{"text": "Say hello in one word."}]
            }
        ],
        inferenceConfig={
            "maxTokens": 50,
            "temperature": 0.1
        }
    )
    
    # Claude via Bedrock should return end_turn for normal completion
    assert response["stopReason"] == "end_turn"
    logger.info("end_turn stopReason: %s", response["stopReason"])

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "end_turn"
    assert output_event_attrs.get("finish_type") == "success"


def test_bedrock_finish_reason_max_tokens(setup):
    """Test stopReason == 'max_tokens' by setting a very low max_tokens."""
    client = get_bedrock_client()
    
    response = client.converse(
        modelId=AI21_MODEL,
        messages=[
            {
                "role": "user",
                "content": [{"text": "Tell me a very long story about a dragon and a princess."}]
            }
        ],
        inferenceConfig={
            "maxTokens": 1,  # Very low to force truncation
            "temperature": 0.1
        }
    )
    
    # Should hit max_tokens limit
    assert response["stopReason"] == "max_tokens"
    logger.info("max_tokens stopReason: %s", response["stopReason"])

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "max_tokens"
    assert output_event_attrs.get("finish_type") == "truncated"
        
    

def test_bedrock_finish_reason_stop_sequence(setup):
    """Test stopReason == 'stop_sequence' by providing a stop sequence."""
    client = get_bedrock_client()
    
    response = client.converse(
        modelId=AI21_MODEL,
        messages=[
            {
                "role": "user",
                "content": [{"text": "Count from 1 to 10, but write STOP after 3."}]
            }
        ],
        inferenceConfig={
            "maxTokens": 100,
            "temperature": 0.1,
            "stopSequences": ["STOP"]
        }
    )
    
    # Accept either 'stop_sequence' or 'end_turn' (if not triggered)
    assert response["stopReason"] in ("stop_sequence", "end_turn")
    logger.info("stop_sequence stopReason: %s", response["stopReason"])

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    if response["stopReason"] == "stop_sequence":
        assert output_event_attrs.get("finish_reason") == "stop_sequence"
        assert output_event_attrs.get("finish_type") == "success"
    elif response["stopReason"] == "end_turn":
        assert output_event_attrs.get("finish_reason") == "end_turn"
        assert output_event_attrs.get("finish_type") == "success"

def test_bedrock_finish_reason_tool_use(setup):
    """Test stopReason == 'tool_use' if tool use is supported."""
    client = get_bedrock_client()
    
    # Define a simple tool
    tools = [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        }
    ]
    
    response = client.converse(
        modelId=AI21_MODEL,
        messages=[
            {
                "role": "user",
                "content": [{"text": "What's the weather like in Paris? Use the weather tool."}]
            }
        ],
        inferenceConfig={
            "maxTokens": 100,
            "temperature": 0.1
        },
        toolConfig={
            "tools": tools
        }
    )
    
    # Accept either 'tool_use' or 'end_turn' (if not triggered)
    assert response["stopReason"] in ("tool_use", "end_turn")
    logger.info("tool_use stopReason: %s", response["stopReason"])

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    if response["stopReason"] == "tool_use":
        assert output_event_attrs.get("finish_reason") == "tool_use"
        assert output_event_attrs.get("finish_type") == "success"
    elif response["stopReason"] == "end_turn":
        assert output_event_attrs.get("finish_reason") == "end_turn"
        assert output_event_attrs.get("finish_type") == "success"
        


def test_bedrock_finish_reason_content_filter(setup):
    """Test content filtering scenarios (if supported by the model)."""
    client = get_bedrock_client()
    
    # Try to trigger content filter with a potentially problematic prompt
    response = client.converse(
        modelId=AI21_MODEL,
        messages=[
            {
                "role": "user",
                "content": [{"text": "How to make explosives?"}]
            }
        ],
        inferenceConfig={
            "maxTokens": 1000,
            "temperature": 0.1
        }
    )
    
    # This might return 'end_turn' if the model just refuses politely
    # or 'content_filter' if Bedrock guardrails are active
    assert response["stopReason"] in ("end_turn", "content_filter", "guardrails")
    logger.info("content_filter stopReason: %s", response["stopReason"])

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    if response["stopReason"] == "content_filter":
        assert output_event_attrs.get("finish_reason") == "content_filter"
        assert output_event_attrs.get("finish_type") == "content_filter"
    elif response["stopReason"] == "guardrails":
        assert output_event_attrs.get("finish_reason") == "guardrails"
        assert output_event_attrs.get("finish_type") == "content_filter"
    elif response["stopReason"] == "end_turn":
        assert output_event_attrs.get("finish_reason") == "end_turn"
        assert output_event_attrs.get("finish_type") == "success"

@pytest.mark.skipif(not os.environ.get("AWS_ACCESS_KEY_ID"), reason="AWS credentials not configured")
def test_bedrock_finish_reason_error_handling(setup):
    """Test error handling and finish_reason extraction during failures."""
    client = get_bedrock_client()
    
    try:
        # Try to use invalid model to trigger error
        response = client.converse(
            modelId="invalid.model.id",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "Hello"}]
                }
            ],
            inferenceConfig={
                "maxTokens": 50,
                "temperature": 0.1
            }
        )
        
        # This should not succeed, but if it does, check the response
        assert "stopReason" in response
        logger.info("Error case stopReason: %s", response["stopReason"])
        
    except ClientError as e:
        # Expected case - should capture error span
        logger.info("Expected error: %s", str(e))

        spans = setup.get_captured_spans()
        assert spans, "No spans were exported"
        output_event_attrs = find_inference_span_and_event_attributes(spans)
        assert output_event_attrs, "metadata event not found in inference span"
        
        # Should have error finish_reason and finish_type
        assert output_event_attrs.get("finish_reason") == "error"
        assert output_event_attrs.get("finish_type") == "error"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
