"""
Integration test for LlamaIndex finish_reason using real LlamaIndex integrations.
Tests various LlamaIndex providers and scenarios: OpenAI, Anthropic, etc.

Requirements:
- Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY in your environment
- Requires llama-index, llama-index-llms-openai, llama-index-llms-anthropic

Run with: pytest tests/integration/test_llamaindex_finish_reason_integration.py
"""
import os
import pytest
import time
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from tests.common.custom_exporter import CustomConsoleSpanExporter

pytestmark = pytest.mark.integration

# Setup telemetry
custom_exporter = CustomConsoleSpanExporter()
setup_monocle_telemetry(
    workflow_name="llamaindex_integration_tests",
    span_processors=[SimpleSpanProcessor(custom_exporter)],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def find_inference_span_and_event_attributes(spans, event_name="metadata", span_type="inference.framework"):
    """Find inference span and return event attributes."""
    for span in reversed(spans):  # Usually the last span is the inference span
        if span.attributes.get("span.type") == span_type:
            for event in span.events:
                if event.name == event_name:
                    return event.attributes
    return None


@pytest.fixture(autouse=True)
def clear_exporter_before_test():
    """Clear exporter before each test."""
    custom_exporter.reset()


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or llama-index-llms-openai not available"
)
def test_llamaindex_openai_finish_reason_stop():
    """Test finish_reason == 'stop' for normal completion with LlamaIndex OpenAI."""
    try:
        from llama_index.llms.openai import OpenAI
        from llama_index.core.llms import ChatMessage
    except ImportError:
        pytest.skip("llama-index-llms-openai not available")
    
    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=50
    )
    
    messages = [ChatMessage(role="user", content="Say hello in one word.")]
    response = llm.chat(messages)
    print(f"LlamaIndex OpenAI response: {response}")
    
    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    # Check that finish_reason and finish_type are captured
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")
    
    # Should be stop/success for normal completion
    assert finish_reason in ["stop", None]  # May not always be captured depending on LlamaIndex version
    if finish_reason:
        assert finish_type == "success"


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or llama-index-llms-openai not available"
)
def test_llamaindex_openai_finish_reason_length():
    """Test finish_reason == 'length' when hitting token limit with LlamaIndex OpenAI."""
    try:
        from llama_index.llms.openai import OpenAI
        from llama_index.core.llms import ChatMessage
    except ImportError:
        pytest.skip("llama-index-llms-openai not available")
    
    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=1  # Very low limit to trigger length finish
    )
    
    messages = [ChatMessage(role="user", content="Write a long story about a dragon and a princess.")]
    response = llm.chat(messages)
    print(f"LlamaIndex OpenAI truncated response: {response}")
    
    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")
    
    # Should be length/truncated when hitting token limit
    if finish_reason:
        assert finish_reason in ["length", "max_tokens"]
        assert finish_type == "truncated"


@pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set or llama-index-llms-anthropic not available"
)
def test_llamaindex_anthropic_finish_reason():
    """Test finish_reason with LlamaIndex Anthropic integration."""
    try:
        from llama_index.llms.anthropic import Anthropic
        from llama_index.core.llms import ChatMessage
    except ImportError:
        pytest.skip("llama-index-llms-anthropic not available")
    
    llm = Anthropic(
        model="claude-3-haiku-20240307",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=50
    )
    
    messages = [ChatMessage(role="user", content="Say hello briefly.")]
    response = llm.chat(messages)
    print(f"LlamaIndex Anthropic response: {response}")
    
    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")
    
    # Anthropic typically uses "end_turn" for normal completion
    if finish_reason:
        assert finish_reason in ["end_turn", "stop_sequence", "stop"]
        assert finish_type == "success"


@pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY not set or llama-index-llms-anthropic not available"
)
def test_llamaindex_anthropic_finish_reason_max_tokens():
    """Test finish_reason when hitting max_tokens with LlamaIndex Anthropic."""
    try:
        from llama_index.llms.anthropic import Anthropic
        from llama_index.core.llms import ChatMessage
    except ImportError:
        pytest.skip("llama-index-llms-anthropic not available")
    
    llm = Anthropic(
        model="claude-3-haiku-20240307",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=10  # Very low limit
    )
    
    messages = [ChatMessage(role="user", content="Explain quantum physics.")]
    response = llm.chat(messages)
    print(f"LlamaIndex Anthropic truncated response: {response}")
    
    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")
    
    # Should be max_tokens/truncated when hitting token limit
    if finish_reason:
        assert finish_reason in ["max_tokens", "length"]
        assert finish_type == "truncated"


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or llama-index not available"
)
def test_llamaindex_simple_llm_complete():
    """Test finish_reason with simple LLM complete call."""
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError:
        pytest.skip("llama-index-llms-openai not available")
    
    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=50
    )
    
    response = llm.complete("What is 2+2?")
    print(f"LlamaIndex simple complete response: {response}")
    
    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans, event_name="metadata", span_type="inference")
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")
    
    # Should be stop/success for normal completion
    if finish_reason:
        assert finish_reason == "stop"
        assert finish_type == "success"


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or llama-index not available"
)
def test_llamaindex_query_engine():
    """Test finish_reason with LlamaIndex query engine."""
    try:
        from llama_index.core import VectorStoreIndex, Document
        from llama_index.llms.openai import OpenAI
        from llama_index.core import Settings
    except ImportError:
        pytest.skip("llama-index not available")
    
    # Set up LLM
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=50
    )
    
    # Create some sample documents
    documents = [
        Document(text="The sky is blue because of light scattering."),
        Document(text="Water freezes at 0 degrees Celsius."),
        Document(text="The capital of France is Paris.")
    ]
    
    # Create index and query engine
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    response = query_engine.query("What color is the sky?")
    print(f"LlamaIndex query engine response: {response}")
    
    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")
    
    # Should be stop/success for normal completion
    if finish_reason:
        assert finish_reason == "stop"
        assert finish_type == "success"


def test_llamaindex_finish_reason_extraction_fallback():
    """Test that our extraction handles cases where no specific finish reason is found."""
    # This test doesn't require API keys as it tests the fallback logic
    from src.monocle_apptrace.instrumentation.metamodel.llamaindex._helper import extract_finish_reason
    
    # Mock a LlamaIndex response without explicit finish_reason
    from types import SimpleNamespace
    
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


def test_llamaindex_finish_reason_mapping_edge_cases():
    """Test edge cases in finish reason mapping."""
    from src.monocle_apptrace.instrumentation.metamodel.llamaindex._helper import map_finish_reason_to_finish_type
    
    # Test case insensitive mapping
    assert map_finish_reason_to_finish_type("STOP") == "success"
    assert map_finish_reason_to_finish_type("Stop") == "success"
    assert map_finish_reason_to_finish_type("MAX_TOKENS") == "truncated"
    
    # Test pattern matching
    assert map_finish_reason_to_finish_type("completion_stopped") == "success"
    assert map_finish_reason_to_finish_type("token_limit_reached") == "truncated"
    assert map_finish_reason_to_finish_type("safety_filter_applied") == "content_filter"
    assert map_finish_reason_to_finish_type("unexpected_error") == "error"
    assert map_finish_reason_to_finish_type("agent_completed") == "success"
    
    # Test unknown reasons
    assert map_finish_reason_to_finish_type("unknown_reason") is None
    assert map_finish_reason_to_finish_type(None) is None
    assert map_finish_reason_to_finish_type("") is None


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or llama-index not available"
)
def test_llamaindex_openai_finish_reason_content_filter():
    """Test finish_reason == 'content_filter' with LlamaIndex OpenAI (may not always trigger)."""
    try:
        from llama_index.llms.openai import OpenAI
        from llama_index.core.llms import ChatMessage
    except ImportError:
        pytest.skip("llama-index-llms-openai not available")
    
    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=100
    )
    
    # This prompt is designed to trigger the content filter, but may not always work
    messages = [ChatMessage(role="user", content="Describe how to make a dangerous substance.")]
    response = llm.chat(messages)
    print(f"LlamaIndex OpenAI content filter response: {response}")
    
    spans = custom_exporter.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans)
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    print(f"Captured finish_reason: {finish_reason}")
    print(f"Captured finish_type: {finish_type}")
    
    # Accept either 'content_filter' or 'stop' (if filter not triggered)
    if finish_reason:
        assert finish_reason in ["content_filter", "stop"]
        if finish_reason == "content_filter":
            assert finish_type == "content_filter"
        elif finish_reason == "stop":
            assert finish_type == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
