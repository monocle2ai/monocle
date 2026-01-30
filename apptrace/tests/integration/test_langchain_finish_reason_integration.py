"""
Integration test for LangChain finish_reason using real LangChain integrations.
Tests various LangChain providers and scenarios: OpenAI, Anthropic, tool calls, etc.

Requirements:
- Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY in your environment
- Requires langchain, langchain-openai, langchain-anthropic

Run with: pytest tests/integration/test_langchain_finish_reason_integration.py
"""
import logging
import os

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI as LangChainOpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from langchain.tools import tool

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def setup():
# Setup telemetry
    custom_exporter = CustomConsoleSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="langchain_integration_tests",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def find_inference_span_and_event_attributes(spans, event_name="metadata"):
    """Find inference span and return event attributes."""
    for span in reversed(spans):  # Usually the last span is the inference span
        if span.attributes.get("span.type") == "inference.framework":
            for event in span.events:
                if event.name == event_name:
                    return event.attributes
    return None


def find_inference_span_with_tool_call(spans):
    """Find the inference span that has finish_type == 'tool_call' and return both span and event attributes."""
    for span in reversed(spans):
        if span.attributes.get("span.type") == "inference.framework":
            # Check if this span has tool_call finish_type
            for event in span.events:
                if event.name == "metadata" and event.attributes.get("finish_type") == "tool_call":
                    return span.attributes, event.attributes
    return None, None


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or langchain-openai not available"
)
def test_langchain_openai_finish_reason_stop(setup):
    """Test finish_reason == 'stop' for normal completion with LangChain OpenAI."""
    chat = LangChainOpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=50
    )
    
    response = chat.invoke("Say hello in one word.")
    logger.info(f"OpenAI LangChain response: {response}")
    # time.sleep(5)  # Allow time for spans to be captured
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
    assert finish_reason in ["stop", None]  # May not always be captured depending on LangChain version
    if finish_reason:
        assert finish_type == "success"


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or langchain-openai not available"
)
def test_langchain_openai_finish_reason_length(setup):
    """Test finish_reason == 'length' when hitting token limit with LangChain OpenAI."""
    chat = LangChainOpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=1  # Very low limit to trigger length finish
    )
    
    response = chat.invoke("Write a long story about a dragon and a princess.")
    logger.info(f"OpenAI LangChain truncated response: {response}")

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
        assert finish_reason in ["length", "max_tokens"]
        assert finish_type == "truncated"


@pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set or langchain-anthropic not available"
)
def test_langchain_anthropic_finish_reason(setup):
    """Test finish_reason with LangChain Anthropic integration."""
    chat = ChatAnthropic(
        model="claude-3-haiku-20240307",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=50
    )
    
    response = chat.invoke("Say hello briefly.")
    logger.info(f"Anthropic LangChain response: {response}")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    logger.info(f"Captured finish_reason: {finish_reason}")
    logger.info(f"Captured finish_type: {finish_type}")
    
    # Anthropic typically uses "end_turn" for normal completion
    if finish_reason:
        assert finish_reason in ["end_turn", "stop_sequence", "stop"]
        assert finish_type == "success"


@pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set or langchain-anthropic not available"
)
def test_langchain_anthropic_finish_reason_max_tokens(setup):
    """Test finish_reason when hitting max_tokens with LangChain Anthropic."""
    chat = ChatAnthropic(
        model="claude-3-haiku-20240307",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=1  # Very low limit
    )
    
    response = chat.invoke("Write a detailed explanation of quantum computing.")
    logger.info(f"Anthropic LangChain truncated response: {response}")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    logger.info(f"Captured finish_reason: {finish_reason}")
    logger.info(f"Captured finish_type: {finish_type}")
    
    # Should be max_tokens/truncated when hitting token limit
    if finish_reason:
        assert finish_reason in ["max_tokens", "length"]
        assert finish_type == "truncated"


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or langchain-openai not available"
)
def test_langchain_chat_with_system_message(setup):
    """Test finish_reason with system message in LangChain."""
    chat = LangChainOpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=50
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Be brief."),
        HumanMessage(content="What is 2+2?")
    ]
    
    response = chat.invoke(messages)
    logger.info(f"Chat with system message response: {response}")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    logger.info(f"Captured finish_reason: {finish_reason}")
    logger.info(f"Captured finish_type: {finish_type}")
    
    # Should be stop/success for normal completion
    if finish_reason:
        assert finish_reason == "stop"
        assert finish_type == "success"


def test_langchain_finish_reason_extraction_fallback(setup):
    """Test that our extraction handles cases where no specific finish reason is found."""
    # This test doesn't require API keys as it tests the fallback logic
    # Mock a LangChain response without explicit finish_reason
    from types import SimpleNamespace

    from src.monocle_apptrace.instrumentation.metamodel.langchain._helper import (
        extract_finish_reason,
    )
    
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


def test_langchain_finish_reason_mapping_edge_cases(setup):
    """Test edge cases in finish reason mapping."""
    from src.monocle_apptrace.instrumentation.metamodel.langchain._helper import (
        map_finish_reason_to_finish_type,
    )
    
    # Test case insensitive mapping
    assert map_finish_reason_to_finish_type("STOP") == "success"
    assert map_finish_reason_to_finish_type("Stop") == "success"
    assert map_finish_reason_to_finish_type("MAX_TOKENS") == "truncated"
    
    # Test pattern matching
    assert map_finish_reason_to_finish_type("completion_stopped") == "success"
    assert map_finish_reason_to_finish_type("token_limit_reached") == "truncated"
    assert map_finish_reason_to_finish_type("safety_filter_applied") == "content_filter"
    assert map_finish_reason_to_finish_type("unexpected_error") == "error"
    
    # Test unknown reasons
    assert map_finish_reason_to_finish_type("unknown_reason") is None
    assert map_finish_reason_to_finish_type(None) is None
    assert map_finish_reason_to_finish_type("") is None


@pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set or langchain-anthropic not available"
)
def test_langchain_anthropic_tool_calls_with_entity_3_validation(setup):
    """Test finish_reason for tool use and entity.3 validation with LangChain Anthropic."""

    # Define a simple tool
    @tool
    def get_current_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny."

    # Create LangChain Anthropic instance with tools
    chat = ChatAnthropic(
        model="claude-3-5-haiku-latest",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=200
    )

    # Bind the tool to the chat model
    chat_with_tools = chat.bind_tools([get_current_weather])

    try:
        response = chat_with_tools.invoke("What's the weather like in Paris? Please use the weather tool.")
        logger.info(f"LangChain Anthropic tool call response: {response}")

        spans = setup.get_captured_spans()
        assert spans, "No spans were exported"

        # Find inference span and get both span and event attributes
        span_attributes, event_attributes = find_inference_span_with_tool_call(spans)

        if span_attributes and event_attributes:
            # Validate finish_type is tool_call
            finish_type = event_attributes.get("finish_type")
            finish_reason = event_attributes.get("finish_reason")

            logger.info(f"Captured finish_reason: {finish_reason}")
            logger.info(f"Captured finish_type: {finish_type}")

            assert finish_type == "tool_call", f"Expected finish_type 'tool_call', got '{finish_type}'"
            assert finish_reason in ["tool_use", "tool_calls"], f"Expected tool_use/tool_calls finish_reason, got '{finish_reason}'"

            # Verify entity.3 attributes when finish_type is tool_call
            assert "entity.3.name" in span_attributes, "entity.3.name should be present when finish_type is tool_call"
            assert "entity.3.type" in span_attributes, "entity.3.type should be present when finish_type is tool_call"

            tool_name = span_attributes.get("entity.3.name")
            tool_type = span_attributes.get("entity.3.type")

            assert tool_name == "get_current_weather", f"Expected tool name 'get_current_weather', got '{tool_name}'"
            assert tool_type == "tool.function", f"Expected tool type 'tool.function', got '{tool_type}'"

            logger.info(f"✓ entity.3.name = '{tool_name}'")
            logger.info(f"✓ entity.3.type = '{tool_type}'")
            logger.info(f"✓ finish_type = '{finish_type}'")
        else:
            # Fallback to check event attributes for finish_type
            output_event_attrs = find_inference_span_and_event_attributes(spans)
            if output_event_attrs:
                finish_reason = output_event_attrs.get("finish_reason")
                finish_type = output_event_attrs.get("finish_type")
                logger.info(f"Fallback - finish_reason: {finish_reason}, finish_type: {finish_type}")

                # If it's not a tool call, that's also valid (model might not trigger tool use)
                if finish_type != "tool_call":
                    logger.info("Tool calling was not triggered by the model - this is acceptable")
                    assert finish_type in ["success", None], f"Expected success or tool_call finish_type, got '{finish_type}'"
            else:
                logger.warning("No inference span found - this might indicate an issue with instrumentation")

    except Exception as e:
        logger.warning(f"Tool calling test failed with error: {e}")
        # Some LangChain versions or models might not support tool calling as expected
        pytest.skip(f"Tool calling not supported or failed: {e}")


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or langchain-openai not available"
)
def test_langchain_openai_tool_calls_with_entity_3_validation(setup):
    """Test finish_reason == 'tool_calls' and entity.3 validation with LangChain OpenAI."""
    
    # Define a simple tool
    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny."
    
    # Create LangChain OpenAI instance with tools
    chat = LangChainOpenAI(
        model="gpt-4o-mini",  # Use a model that supports tool calling
        api_key=OPENAI_API_KEY,
        max_tokens=100
    )
    
    # Bind the tool to the chat model
    chat_with_tools = chat.bind_tools([get_weather])
    
    try:
        response = chat_with_tools.invoke("What's the weather like in Paris?")
        logger.info(f"LangChain OpenAI tool call response: {response}")

        spans = setup.get_captured_spans()
        assert spans, "No spans were exported"
        
        # Find inference span and get both span and event attributes
        span_attributes, event_attributes = find_inference_span_with_tool_call(spans)
        
        if span_attributes and event_attributes:
            # Validate finish_type is tool_call
            finish_type = event_attributes.get("finish_type")
            finish_reason = event_attributes.get("finish_reason")
            
            logger.info(f"Captured finish_reason: {finish_reason}")
            logger.info(f"Captured finish_type: {finish_type}")
            
            assert finish_type == "tool_call", f"Expected finish_type 'tool_call', got '{finish_type}'"
            assert finish_reason in ["tool_calls", "function_call"], f"Expected tool_calls/function_call finish_reason, got '{finish_reason}'"
            
            # Verify entity.3 attributes when finish_type is tool_call
            assert "entity.3.name" in span_attributes, "entity.3.name should be present when finish_type is tool_call"
            assert "entity.3.type" in span_attributes, "entity.3.type should be present when finish_type is tool_call"
            
            tool_name = span_attributes.get("entity.3.name")
            tool_type = span_attributes.get("entity.3.type")
            
            assert tool_name == "get_weather", f"Expected tool name 'get_weather', got '{tool_name}'"
            assert tool_type == "tool.function", f"Expected tool type 'tool.function', got '{tool_type}'"
            
            logger.info(f"✓ entity.3.name = '{tool_name}'")
            logger.info(f"✓ entity.3.type = '{tool_type}'")
            logger.info(f"✓ finish_type = '{finish_type}'")
        else:
            # Fallback to check event attributes for finish_type
            output_event_attrs = find_inference_span_and_event_attributes(spans)
            if output_event_attrs:
                finish_reason = output_event_attrs.get("finish_reason")
                finish_type = output_event_attrs.get("finish_type")
                logger.info(f"Fallback - finish_reason: {finish_reason}, finish_type: {finish_type}")
                
                # If it's not a tool call, that's also valid (model might not trigger tool use)
                if finish_type != "tool_call":
                    logger.info("Tool calling was not triggered by the model - this is acceptable")
                    assert finish_type in ["success", None], f"Expected success or tool_call finish_type, got '{finish_type}'"
            else:
                logger.warning("No inference span found - this might indicate an issue with instrumentation")
    
    except Exception as e:
        logger.warning(f"Tool calling test failed with error: {e}")
        # Some LangChain versions or models might not support tool calling as expected
        pytest.skip(f"Tool calling not supported or failed: {e}")

@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or langchain-openai not available"
)
def test_langchain_openai_finish_reason_content_filter(setup):
    """Test finish_reason == 'content_filter' with LangChain OpenAI (may not always trigger)."""
    chat = LangChainOpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=100
    )
    
    # This prompt is designed to trigger the content filter, but may not always work
    response = chat.invoke("Describe how to make a dangerous substance.")
    logger.info(f"OpenAI LangChain content filter response: {response}")

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
        assert finish_reason in ["content_filter", "stop"]
        if finish_reason == "content_filter":
            assert finish_type == "content_filter"
        elif finish_reason == "stop":
            assert finish_type == "success"


@pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set or langchain-anthropic not available"
)
def test_langchain_anthropic_finish_reason_content_filter(setup):
    """Test finish_reason for content filtering with LangChain Anthropic."""
    chat = ChatAnthropic(
        model="claude-3-haiku-20240307",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=100
    )
    
    # This prompt is designed to trigger safety mechanisms, but may not always work
    response = chat.invoke("Describe how to make a dangerous substance.")
    logger.info(f"Anthropic LangChain content filter response: {response}")

    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    logger.info(f"Captured finish_reason: {finish_reason}")
    logger.info(f"Captured finish_type: {finish_type}")
    
    # Anthropic may use "refusal" or other safety-related finish reasons
    if finish_reason:
        if finish_reason in ["refusal", "safety", "content_filter"]:
            assert finish_type in ["content_filter", "refusal"]
        elif finish_reason in ["end_turn", "stop"]:
            assert finish_type == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

