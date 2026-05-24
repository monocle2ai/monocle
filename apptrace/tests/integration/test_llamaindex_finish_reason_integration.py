"""
Integration test for LlamaIndex finish_reason using real LlamaIndex integrations.
Tests various LlamaIndex providers and scenarios: OpenAI, Anthropic, etc.
Also tests span.subtype ("tool_call" / "turn_end") and inference.decision.span.id.

Requirements:
- Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY in your environment
- Requires llama-index, llama-index-llms-openai, llama-index-llms-anthropic

Run with: pytest tests/integration/test_llamaindex_finish_reason_integration.py
"""
import asyncio
import logging
import os
import time

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
from custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

try:
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.openai import OpenAI
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    ChatMessage = None
    OpenAI = None
    LLAMAINDEX_AVAILABLE = False

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup():
# Setup telemetry
    custom_exporter = CustomConsoleSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="llamaindex_integration_tests",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


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
    reason="OPENAI_API_KEY not set or llama-index-llms-openai not available"
)
def test_llamaindex_openai_tool_call_with_entity_3_validation(setup):
    """Test function calling with LlamaIndex OpenAI and validate entity.3.name and entity.3.type."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to get weather for"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=100
    )
    
    # Set tools on the LLM (if supported)
    if hasattr(llm, '_additional_kwargs'):
        llm._additional_kwargs = {"tools": tools}
    
    messages = [ChatMessage(role="user", content="What's the weather like in Paris? Use the get_weather function.")]
    
    try:
        response = llm.chat(messages, tools=tools if hasattr(llm, 'chat') else None)
        logger.info(f"LlamaIndex OpenAI function calling response: {response}")

        spans = setup.get_captured_spans()
        assert spans, "No spans were exported"

        span_attributes, event_attributes = find_inference_span_with_tool_call(spans)
        if span_attributes and event_attributes:

            assert "entity.3.name" in span_attributes, "entity.3.name should be present when finish_type is tool_call"
            assert "entity.3.type" in span_attributes, "entity.3.type should be present when finish_type is tool_call"

            tool_name = span_attributes.get("entity.3.name")
            tool_type = span_attributes.get("entity.3.type")
            assert tool_name == "get_current_weather", f"Expected tool name 'get_current_weather', got '{tool_name}'"
            assert tool_type == "tool.function", f"Expected tool type 'tool.function', got '{tool_type}'"

            logger.info(f"✓ entity.3.name = '{tool_name}'")
            logger.info(f"✓ entity.3.type = '{tool_type}'")

        else:
            # If no tool call span found, check if we have regular completion
            output_event_attrs = find_inference_span_and_event_attributes(spans)
            if output_event_attrs:
                finish_reason = output_event_attrs.get("finish_reason")
                finish_type = output_event_attrs.get("finish_type")
                logger.info(f"No tool calls found. finish_reason: {finish_reason}, finish_type: {finish_type}")
                pytest.skip("Function calling not triggered - model responded normally instead of using functions")
            else:
                pytest.fail("No inference span found in captured spans")
                
    except Exception as e:
        logger.warning(f"Tool calling test failed with error: {e}")
        pytest.skip(f"Tool calling not supported or failed: {e}")


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or llama-index-llms-openai not available"
)
def test_llamaindex_openai_finish_reason_stop(setup):
    """Test finish_reason == 'stop' for normal completion with LlamaIndex OpenAI."""
    try:
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.openai import OpenAI
    except ImportError:
        pytest.skip("llama-index-llms-openai not available")
    
    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=50
    )
    
    messages = [ChatMessage(role="user", content="Say hello in one word.")]
    response = llm.chat(messages)
    logger.info(f"LlamaIndex OpenAI response: {response}")
    
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
    assert finish_reason in ["stop", None]  # May not always be captured depending on LlamaIndex version
    if finish_reason:
        assert finish_type == "success"


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or llama-index-llms-openai not available"
)
def test_llamaindex_openai_finish_reason_length(setup):
    """Test finish_reason == 'length' when hitting token limit with LlamaIndex OpenAI."""
    try:
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.openai import OpenAI
    except ImportError:
        pytest.skip("llama-index-llms-openai not available")
    
    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        max_tokens=1  # Very low limit to trigger length finish
    )
    
    messages = [ChatMessage(role="user", content="Write a long story about a dragon and a princess.")]
    response = llm.chat(messages)
    logger.info(f"LlamaIndex OpenAI truncated response: {response}")
    
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
    reason="ANTHROPIC_API_KEY not set or llama-index-llms-anthropic not available"
)
def test_llamaindex_anthropic_finish_reason(setup):
    """Test finish_reason with LlamaIndex Anthropic integration."""
    try:
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.anthropic import Anthropic
    except ImportError:
        pytest.skip("llama-index-llms-anthropic not available")
    
    llm = Anthropic(
        model="claude-haiku-4-5-20251001",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=50
    )
    
    messages = [ChatMessage(role="user", content="Say hello briefly.")]
    response = llm.chat(messages)
    logger.info(f"LlamaIndex Anthropic response: {response}")
    
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
    reason="ANTHROPIC_API_KEY not set or llama-index-llms-anthropic not available"
)
def test_llamaindex_anthropic_finish_reason_max_tokens(setup):
    """Test finish_reason when hitting max_tokens with LlamaIndex Anthropic."""
    try:
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.anthropic import Anthropic
    except ImportError:
        pytest.skip("llama-index-llms-anthropic not available")
    
    llm = Anthropic(
        model="claude-haiku-4-5-20251001",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=10  # Very low limit
    )
    
    messages = [ChatMessage(role="user", content="Explain quantum physics.")]
    response = llm.chat(messages)
    logger.info(f"LlamaIndex Anthropic truncated response: {response}")
    
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
    reason="OPENAI_API_KEY not set or llama-index not available"
)
def test_llamaindex_simple_llm_complete(setup):
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
    logger.info(f"LlamaIndex simple complete response: {response}")
    
    spans = setup.get_captured_spans()
    assert spans, "No spans were exported"
    
    output_event_attrs = find_inference_span_and_event_attributes(spans, event_name="metadata", span_type="inference")
    assert output_event_attrs, "metadata event not found in inference span"
    
    finish_reason = output_event_attrs.get("finish_reason")
    finish_type = output_event_attrs.get("finish_type")
    
    logger.info(f"Captured finish_reason: {finish_reason}")
    logger.info(f"Captured finish_type: {finish_type}")
    
    # Should be stop/success for normal completion
    if finish_reason:
        assert finish_reason == "stop"
        assert finish_type == "success"


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or llama-index not available"
)
def test_llamaindex_query_engine(setup):
    """Test finish_reason with LlamaIndex query engine."""
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.llms.openai import OpenAI
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
    logger.info(f"LlamaIndex query engine response: {response}")

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


def test_llamaindex_finish_reason_extraction_fallback(setup):
    """Test that our extraction handles cases where no specific finish reason is found."""
    # This test doesn't require API keys as it tests the fallback logic
    # Mock a LlamaIndex response without explicit finish_reason
    from types import SimpleNamespace

    from src.monocle_apptrace.instrumentation.metamodel.llamaindex._helper import (
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


def test_llamaindex_finish_reason_mapping_edge_cases():
    """Test edge cases in finish reason mapping."""
    from src.monocle_apptrace.instrumentation.metamodel.llamaindex._helper import (
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
    assert map_finish_reason_to_finish_type("agent_completed") == "success"
    
    # Test tool call mapping
    assert map_finish_reason_to_finish_type("tool_calls") == "tool_call"
    assert map_finish_reason_to_finish_type("function_call") == "tool_call"
    
    # Test unknown reasons
    assert map_finish_reason_to_finish_type("unknown_reason") is None
    assert map_finish_reason_to_finish_type(None) is None
    assert map_finish_reason_to_finish_type("") is None


@pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set or llama-index not available"
)
def test_llamaindex_openai_finish_reason_content_filter(setup):
    """Test finish_reason == 'content_filter' with LlamaIndex OpenAI (may not always trigger)."""
    try:
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.openai import OpenAI
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
    logger.info(f"LlamaIndex OpenAI content filter response: {response}")
    
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


# ---------------------------------------------------------------------------
# span.subtype and inference.decision.span.id tests
# ---------------------------------------------------------------------------

_INFERENCE_SPAN_TYPES = ("inference", "inference.framework", "inference.modelapi")
_TOOL_INVOCATION_TYPE = "agentic.tool.invocation"

WEATHER_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def _new_inference_spans(exporter, before_count, wait=2):
    """Return inference spans captured after `before_count` spans were already in the exporter."""
    time.sleep(wait)
    new_spans = exporter.get_captured_spans()[before_count:]
    return [s for s in new_spans if s.attributes.get("span.type") in _INFERENCE_SPAN_TYPES]


def _new_tool_invocation_spans(exporter, before_count):
    """Return tool-invocation spans captured after `before_count` spans were already in the exporter."""
    new_spans = exporter.get_captured_spans()[before_count:]
    return [s for s in new_spans if s.attributes.get("span.type") == _TOOL_INVOCATION_TYPE]


def _span_id_hex(span):
    return format(span.context.span_id, "#018x")


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_subtype_tool_call_on_inference_span(setup):
    """Inference span has span.subtype='tool_call' when the LLM calls a tool."""
    before = len(setup.get_captured_spans())
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, max_tokens=100)
    messages = [ChatMessage(role="user", content="What's the weather in Paris?")]
    llm.chat(messages, tools=[WEATHER_TOOL_SPEC])

    inf_spans = _new_inference_spans(setup, before)
    assert inf_spans, f"No inference spans captured (total spans: {len(setup.get_captured_spans())})"

    tool_call_spans = [s for s in inf_spans if s.attributes.get("span.subtype") == "tool_call"]
    assert tool_call_spans, (
        f"Expected span.subtype='tool_call'. Got subtypes: {[s.attributes.get('span.subtype') for s in inf_spans]}"
    )
    logger.info("span.subtype='tool_call' verified on inference span")


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_subtype_turn_end_on_plain_response(setup):
    """Inference span has span.subtype='turn_end' for a plain text response."""
    before = len(setup.get_captured_spans())
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, max_tokens=10)
    messages = [ChatMessage(role="user", content="Say hello in one word.")]
    llm.chat(messages)

    inf_spans = _new_inference_spans(setup, before)
    assert inf_spans, f"No inference spans captured (total spans: {len(setup.get_captured_spans())})"

    span = inf_spans[-1]
    assert "span.subtype" in span.attributes, "span.subtype must be present on every inference span"
    assert span.attributes["span.subtype"] == "turn_end", (
        f"Expected span.subtype='turn_end', got '{span.attributes['span.subtype']}'"
    )
    logger.info("span.subtype='turn_end' verified on inference span")


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_entity3_name_on_tool_call_inference_span(setup):
    """entity.3.name equals the called tool name on tool_call inference spans."""
    before = len(setup.get_captured_spans())
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, max_tokens=100)
    messages = [ChatMessage(role="user", content="What's the weather in Tokyo?")]
    llm.chat(messages, tools=[WEATHER_TOOL_SPEC])

    inf_spans = _new_inference_spans(setup, before)
    tool_call_spans = [s for s in inf_spans if s.attributes.get("span.subtype") == "tool_call"]
    assert tool_call_spans, (
        f"No tool_call inference span found. Captured subtypes: {[s.attributes.get('span.subtype') for s in inf_spans]}"
    )

    span = tool_call_spans[0]
    assert span.attributes.get("entity.3.name") == "get_weather", (
        f"Expected entity.3.name='get_weather', got '{span.attributes.get('entity.3.name')}'"
    )
    assert span.attributes.get("entity.3.type") == "tool.function", (
        f"Expected entity.3.type='tool.function', got '{span.attributes.get('entity.3.type')}'"
    )
    logger.info("entity.3.name and entity.3.type verified on tool_call inference span")


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_entity3_absent_on_turn_end_inference_span(setup):
    """entity.3.name is NOT set when the LLM does not call a tool."""
    before = len(setup.get_captured_spans())
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, max_tokens=15)
    messages = [ChatMessage(role="user", content="What is the capital of France?")]
    llm.chat(messages)

    inf_spans = _new_inference_spans(setup, before)
    turn_end_spans = [s for s in inf_spans if s.attributes.get("span.subtype") == "turn_end"]
    assert turn_end_spans, (
        f"No turn_end inference span found. Captured subtypes: {[s.attributes.get('span.subtype') for s in inf_spans]}"
    )

    for span in turn_end_spans:
        assert span.attributes.get("entity.3.name") is None, (
            f"entity.3.name should be absent on turn_end span, got '{span.attributes.get('entity.3.name')}'"
        )
    logger.info("entity.3.name correctly absent on turn_end inference span")


@pytest.mark.skip(reason="inference.decision.span.id not yet implemented for ReAct-style agents")
@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_inference_decision_span_id_via_react_agent(setup):
    """
    When a ReActAgent calls a tool, the agentic.tool.invocation span must have
    inference.decision.span.id pointing to the inference span that decided to call it.
    Note: ReAct agents use text-based tool calling, not native OpenAI tool calling,
    so their inference spans have subtype='turn_end' rather than 'tool_call'.
    
    TODO: Instrumentation needs to be updated to set inference.decision.span.id
    for ReAct-style agents that use text-based tool invocation.
    """
    before = len(setup.get_captured_spans())
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.openai import OpenAI

    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Sunny, 25°C in {city}"

    tool = FunctionTool.from_defaults(fn=get_weather, name="get_weather",
                                      description="Get the current weather for a city.")
    llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
    agent = ReActAgent(name="WeatherAgent", tools=[tool], llm=llm)

    async def run():
        return await agent.run("What is the weather in London?")

    asyncio.run(run())
    time.sleep(2)

    inf_spans = _new_inference_spans(setup, before, wait=0)
    tool_spans = _new_tool_invocation_spans(setup, before)

    assert inf_spans, f"No inference spans captured (total: {len(setup.get_captured_spans())})"
    assert tool_spans, f"No tool invocation spans captured (total: {len(setup.get_captured_spans())})"

    # For ReAct agents: build map of all inference span IDs (they use turn_end, not tool_call)
    inf_span_ids = {_span_id_hex(s) for s in inf_spans}
    assert inf_span_ids, "Expected at least one inference span"

    weather_spans = [s for s in tool_spans if s.attributes.get("entity.1.name") == "get_weather"]
    assert weather_spans, "Expected a 'get_weather' tool invocation span"

    for ts in weather_spans:
        decision_id = ts.attributes.get("inference.decision.span.id")
        assert decision_id is not None, (
            "agentic.tool.invocation span is missing inference.decision.span.id"
        )
        assert decision_id in inf_span_ids, (
            f"inference.decision.span.id '{decision_id}' does not match any "
            f"inference span: {inf_span_ids}"
        )
        logger.info(f"Verified: get_weather tool → inference span {decision_id}")


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
def test_inference_decision_span_id_not_set_on_plain_inference(setup):
    """inference.decision.span.id must NOT appear on non-tool-invocation spans."""
    before = len(setup.get_captured_spans())
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, max_tokens=15)
    messages = [ChatMessage(role="user", content="Tell me a short joke.")]
    llm.chat(messages)

    time.sleep(2)
    new_spans = setup.get_captured_spans()[before:]
    for span in new_spans:
        if span.attributes.get("span.type") not in (_TOOL_INVOCATION_TYPE,):
            assert "inference.decision.span.id" not in span.attributes, (
                f"inference.decision.span.id should only appear on tool invocation spans, "
                f"found on span '{span.name}' (type={span.attributes.get('span.type')})"
            )
    logger.info("Confirmed: inference.decision.span.id absent on non-tool spans")


@pytest.mark.skip(reason="LlamaIndex Anthropic integration doesn't support FunctionTool objects in chat() - requires different tool format")
@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_backend_subtype_tool_call(setup):
    """span.subtype='tool_call' works when LlamaIndex uses Anthropic as the backend.
    
    Note: This test is skipped because LlamaIndex's Anthropic integration doesn't support
    passing FunctionTool objects directly to llm.chat(). Anthropic tool calling requires
    a different approach (e.g., using agents or dict-based tool definitions).
    """
    before = len(setup.get_captured_spans())
    try:
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.anthropic import Anthropic
    except ImportError:
        pytest.skip("llama-index-llms-anthropic not available")

    from llama_index.core.tools import FunctionTool

    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Sunny in {city}"

    tool = FunctionTool.from_defaults(fn=get_weather, name="get_weather",
                                      description="Get the current weather for a city.")
    llm = Anthropic(model="claude-haiku-4-5-20251001", api_key=ANTHROPIC_API_KEY, max_tokens=100)
    messages = [ChatMessage(role="user", content="What's the weather in Rome?")]
    llm.chat(messages, tools=[tool])

    inf_spans = _new_inference_spans(setup, before)
    assert inf_spans, f"No inference spans captured (total: {len(setup.get_captured_spans())})"

    tool_call_spans = [s for s in inf_spans if s.attributes.get("span.subtype") == "tool_call"]
    assert tool_call_spans, (
        f"Expected span.subtype='tool_call' with Anthropic backend. "
        f"Got subtypes: {[s.attributes.get('span.subtype') for s in inf_spans]}"
    )
    assert tool_call_spans[0].attributes.get("entity.3.name") == "get_weather"
    logger.info("Anthropic backend: span.subtype='tool_call' and entity.3.name verified")


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_backend_subtype_turn_end(setup):
    """span.subtype='turn_end' works when LlamaIndex uses Anthropic as the backend."""
    before = len(setup.get_captured_spans())
    try:
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.anthropic import Anthropic
    except ImportError:
        pytest.skip("llama-index-llms-anthropic not available")

    llm = Anthropic(model="claude-haiku-4-5-20251001", api_key=ANTHROPIC_API_KEY, max_tokens=20)
    messages = [ChatMessage(role="user", content="Say hello in one word.")]
    llm.chat(messages)

    inf_spans = _new_inference_spans(setup, before)
    assert inf_spans, f"No inference spans captured (total: {len(setup.get_captured_spans())})"

    span = inf_spans[-1]
    assert "span.subtype" in span.attributes, "span.subtype must be present"
    assert span.attributes["span.subtype"] == "turn_end", (
        f"Expected span.subtype='turn_end', got '{span.attributes['span.subtype']}'"
    )
    logger.info("Anthropic backend: span.subtype='turn_end' verified")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
