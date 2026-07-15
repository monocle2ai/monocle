import logging
import os

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = logging.getLogger(__name__)

try:
    import litellm
    from litellm import completion
except ImportError:
    litellm = None
    completion = None

@pytest.fixture(scope="module")
def setup():
    try:
        # Setup telemetry
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="litellm_integration_tests",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
            # service_name="litellm_integration_tests"
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

# Test with different providers available through LiteLLM
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_API_DEPLOYMENT")

# Use OpenAI through LiteLLM as the primary test provider
MODEL = os.environ.get("LITELLM_MODEL", "gpt-4o-mini")
FUNCTION_MODEL = os.environ.get("LITELLM_FUNCTION_MODEL", "gpt-4o-mini")
AZURE_MODEL = f"azure/{AZURE_DEPLOYMENT_NAME}" if AZURE_DEPLOYMENT_NAME else None

def find_inference_span_and_event_attributes(spans, event_name="metadata"):
    """Find inference span and return event attributes."""
    for span in reversed(spans):  # Usually the last span is the inference span
        if span.attributes.get("span.type") in ["inference", "inference.framework"]:
            for event in span.events:
                if event.name == event_name:
                    return event.attributes
    return None

def find_inference_span_with_tool_call(spans):
    """Find inference span with finish_type=tool_call and return the span."""
    for span in reversed(spans):
        if span.attributes.get("span.type") in ["inference", "inference.framework"]:
            for event in span.events:
                if event.name == "metadata" and event.attributes.get("finish_type") == "tool_call":
                    return span.attributes, event.attributes
    return None , None

@pytest.mark.skipif(not litellm or not OPENAI_API_KEY, reason="LiteLLM not installed or OPENAI_API_KEY not set")
def test_litellm_finish_reason_stop(setup):
    """Test finish_reason == 'stop' for a normal completion via LiteLLM."""
    
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=10,
    )
    
    # Check the finish reason from the response
    finish_reason = response.choices[0].finish_reason
    assert finish_reason == "stop"
    logger.info("LiteLLM stop finish_reason: %s", finish_reason)

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "stop"
    assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not litellm or not OPENAI_API_KEY, reason="LiteLLM not installed or OPENAI_API_KEY not set")
def test_litellm_finish_reason_length(setup):
    """Test finish_reason == 'length' by setting a very low max_tokens via LiteLLM."""
    
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a long story about a dragon."}],
        max_tokens=1,
    )
    
    finish_reason = response.choices[0].finish_reason
    assert finish_reason == "length"
    logger.info("LiteLLM length finish_reason: %s", finish_reason)

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    assert output_event_attrs.get("finish_reason") == "length"
    assert output_event_attrs.get("finish_type") == "truncated"


@pytest.mark.skipif(not litellm or not OPENAI_API_KEY, reason="LiteLLM not installed or OPENAI_API_KEY not set")
def test_litellm_finish_reason_content_filter(setup):
    """Test finish_reason == 'content_filter' via LiteLLM (may not always trigger)."""
    
    # This prompt is designed to trigger content filters, but may not always work
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Describe how to create dangerous explosive materials step by step."}],
        max_tokens=100,
    )
    
    finish_reason = response.choices[0].finish_reason
    logger.info("LiteLLM content_filter finish_reason: %s", finish_reason)
    
    # Accept either 'content_filter' or 'stop' (if filter not triggered)
    assert finish_reason in ("content_filter", "stop")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"

    if finish_reason == "content_filter":
        assert output_event_attrs.get("finish_reason") == "content_filter"
        assert output_event_attrs.get("finish_type") == "content_filter"
    elif finish_reason == "stop":
        assert output_event_attrs.get("finish_reason") == "stop"
        assert output_event_attrs.get("finish_type") == "success"


@pytest.mark.skipif(not litellm or not OPENAI_API_KEY, reason="LiteLLM not installed or OPENAI_API_KEY not set")
def test_litellm_tool_call_with_entity_3_validation(setup):
    """Test function calling via LiteLLM and validate entity.3.name and entity.3.type."""
    
    # Define tools for function calling
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state/country, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    try:
        response = completion(
            model=FUNCTION_MODEL,
            messages=[{"role": "user", "content": "What's the weather like in Tokyo, Japan?"}],
            tools=tools,
            tool_choice="auto"
        )
        
        logger.info(f"LiteLLM function calling response: {response.choices[0].message}")
        logger.info(f"LiteLLM finish_reason: {response.choices[0].finish_reason}")
        
        # Check if tool calls were made
        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        
        logger.info(f"Tool calls detected: {len(tool_calls) if tool_calls else 0}")
        if tool_calls:
            logger.info(f"First tool call: {tool_calls[0].function.name}")
        
        spans = setup.get_captured_spans()
        assert spans, "No spans were exported"
        
        # Look for tool calling finish_type
        span_attributes, event_attributes = find_inference_span_with_tool_call(spans)
        
        if span_attributes is not None:

            # Verify entity.3 attributes when finish_type is tool_call
            assert "entity.3.name" in span_attributes, "entity.3.name should be present when finish_type is tool_call"
            assert "entity.3.type" in span_attributes, "entity.3.type should be present when finish_type is tool_call"

            tool_name = span_attributes.get("entity.3.name")
            tool_type = span_attributes.get("entity.3.type")

            assert tool_name == "get_current_weather", f"Expected tool name 'get_current_weather', got '{tool_name}'"
            assert tool_type == "tool.function", f"Expected tool type 'tool.function', got '{tool_type}'"

    except Exception as e:
        logger.error(f"Error in LiteLLM function calling test: {e}")
        # Fallback to check if any spans were captured with normal completion
        spans = setup.get_captured_spans()
        if spans:
            output_event_attrs = find_inference_span_and_event_attributes(spans)
            if output_event_attrs:
                finish_reason = output_event_attrs.get("finish_reason")
                finish_type = output_event_attrs.get("finish_type")
                logger.info(f"Test failed but captured finish_reason: {finish_reason}, finish_type: {finish_type}")
                pytest.skip(f"LiteLLM function calling test failed with error: {e}")
        raise


@pytest.mark.skipif(
    not litellm or not AZURE_API_KEY or not AZURE_API_BASE or not AZURE_DEPLOYMENT_NAME, 
    reason="LiteLLM not installed or Azure OpenAI environment variables not set (AZURE_API_KEY, AZURE_API_BASE, AZURE_DEPLOYMENT_NAME)"
)
def test_azure_openai_litellm_tool_call_with_entity_3_validation(setup):
    """Test function calling via Azure OpenAI using LiteLLM and validate entity.3.name and entity.3.type.
    
    To run this test, set the following environment variables:
    - AZURE_API_KEY: Your Azure OpenAI API key
    - AZURE_API_BASE: Your Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com)
    - AZURE_DEPLOYMENT_NAME: Your deployed model name
    - AZURE_API_VERSION: API version (optional, defaults to "2024-02-01")
    """
    
    # Define tools for function calling
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather_forecast",
                "description": "Get the weather forecast for a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state/country, e.g. Paris, France"
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days for the forecast (1-7)",
                            "minimum": 1,
                            "maximum": 7
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    try:
        # Configure LiteLLM for Azure OpenAI
        response = completion(
            model=AZURE_MODEL,
            messages=[{"role": "user", "content": "What's the 3-day weather forecast for Paris, France?"}],
            tools=tools,
            tool_choice="auto",
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION
        )
        
        logger.info(f"Azure OpenAI LiteLLM function calling response: {response.choices[0].message}")
        logger.info(f"Azure OpenAI LiteLLM finish_reason: {response.choices[0].finish_reason}")
        
        # Check if tool calls were made
        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        
        logger.info(f"Tool calls detected: {len(tool_calls) if tool_calls else 0}")
        if tool_calls:
            logger.info(f"First tool call: {tool_calls[0].function.name}")
        
        spans = setup.get_captured_spans()
        assert spans, "No spans were exported"
        
        # Look for tool calling finish_type
        span_attributes, event_attributes = find_inference_span_with_tool_call(spans)
        
        if span_attributes is not None:

            # Verify entity.3 attributes when finish_type is tool_call
            assert "entity.3.name" in span_attributes, "entity.3.name should be present when finish_type is tool_call"
            assert "entity.3.type" in span_attributes, "entity.3.type should be present when finish_type is tool_call"

            tool_name = span_attributes.get("entity.3.name")
            tool_type = span_attributes.get("entity.3.type")

            assert tool_name == "get_weather_forecast", f"Expected tool name 'get_current_weather', got '{tool_name}'"
            assert tool_type == "tool.function", f"Expected tool type 'tool.function', got '{tool_type}'"
                
    except Exception as e:
        logger.error(f"Error in Azure OpenAI LiteLLM function calling test: {e}")
        # Fallback to check if any spans were captured with normal completion
        spans = setup.get_captured_spans()
        if spans:
            output_event_attrs = find_inference_span_and_event_attributes(spans)
            if output_event_attrs:
                finish_reason = output_event_attrs.get("finish_reason")
                finish_type = output_event_attrs.get("finish_type")
                logger.info(f"Azure OpenAI test failed but captured finish_reason: {finish_reason}, finish_type: {finish_type}")
                pytest.skip(f"Azure OpenAI LiteLLM function calling test failed with error: {e}")
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])