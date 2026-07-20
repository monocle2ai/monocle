"""HuggingFace Inference **streaming** scenarios via ``huggingface_hub`` SDK.

Each test function is a self-contained, realistic streaming agent behaviour that
produces a *distinct* trace.  They deliberately avoid duplicating the existing
basic streaming test in test_hugging_face_inference.py — every scenario below
adds a behaviour that changes the shape of the emitted trace.

Scenarios in this file:
    1. test_stream_sync_basic ........... sync streaming path
    2. test_stream_async_basic .......... async streaming path
    3. test_stream_long_answer .......... very large streamed response (long-output trace)
    4. test_stream_multi_turn ........... multi-turn conversation, one stream per turn
    5. test_stream_with_history ......... single stream primed with prior conversation history
    6. test_stream_single_tool_call ..... stream -> one tool call -> stream final answer
    7. test_stream_mixed_with_non_stream  a non-streamed call followed by a streamed call
    8. test_stream_cancellation ......... consumer cancels/interrupts the stream mid-flight

All scenarios reuse ``common.stream_helpers`` for Okahu+in-memory wiring and
``common.helpers`` for span verification.
"""

import asyncio
import json
import logging
import os

import pytest
from common.helpers import find_spans_by_type, verify_inference_span
from common.stream_helpers import build_stream_span_processors
from huggingface_hub import InferenceClient, AsyncInferenceClient
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

logger = logging.getLogger(__name__)

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
API_KEY = os.environ.get("HUGGING_FACE_API_KEY", "")


@pytest.fixture(scope="function")
def setup():
    exporter, span_processors = build_stream_span_processors()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="huggingface_streaming_scenarios",
            span_processors=span_processors,
            wrapper_methods=[],
        )
        yield exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# Local tool definitions + helpers
# ---------------------------------------------------------------------------

COFFEE_MENU = {
    "espresso": "A strong and bold coffee shot.",
    "latte": "A smooth coffee with steamed milk.",
    "cappuccino": "A rich coffee with frothy milk foam.",
    "americano": "Espresso with added hot water for a milder taste.",
}


def get_coffee_details(coffee_name: str) -> str:
    """Get the description of a coffee drink by name."""
    return COFFEE_MENU.get(coffee_name.lower(), f"Unknown coffee: {coffee_name}")


COFFEE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_coffee_details",
            "description": "Get the description of a coffee drink by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coffee_name": {
                        "type": "string",
                        "description": "Name of the coffee drink, e.g. 'latte'.",
                    }
                },
                "required": ["coffee_name"],
            },
        },
    },
]


def _collect_hf_stream(stream):
    """Drain a sync HuggingFace stream, returning accumulated text."""
    text_parts = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text_parts.append(chunk.choices[0].delta.content)
    return "".join(text_parts)


async def _acollect_hf_stream(stream):
    """Drain an async HuggingFace stream, returning accumulated text."""
    text_parts = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text_parts.append(chunk.choices[0].delta.content)
    return "".join(text_parts)


def _assert_has_stream_inference(exporter, min_count=1):
    """Verify we have at least min_count inference spans with proper attributes."""
    spans = exporter.get_captured_spans()
    inference_spans = find_spans_by_type(spans, "inference") or find_spans_by_type(
        spans, "inference.framework"
    )
    assert len(inference_spans) >= min_count
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.huggingface",
            model_name=MODEL,
            model_type=f"model.llm.{MODEL}",
            check_metadata=False,
            check_input_output=True,
        )
    return inference_spans


# ---------------------------------------------------------------------------
# 1. Sync streaming basic
# ---------------------------------------------------------------------------
def test_stream_sync_basic(setup):
    """Basic sync streaming: one inference span, text response."""
    if not API_KEY:
        pytest.skip("HUGGING_FACE_API_KEY not set")
    
    client = InferenceClient(api_key=API_KEY)
    
    stream = client.chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in 5 words."},
        ],
        stream=True,
        max_tokens=50,
    )
    
    text = _collect_hf_stream(stream)
    assert len(text) > 0, "Expected non-empty streamed response"
    
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 2. Async streaming basic
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_async_basic(setup):
    """Basic async streaming: one inference span, text response."""
    if not API_KEY:
        pytest.skip("HUGGING_FACE_API_KEY not set")
    
    client = AsyncInferenceClient(api_key=API_KEY)
    
    stream = await client.chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in 5 words."},
        ],
        stream=True,
        max_tokens=50,
    )
    
    text = await _acollect_hf_stream(stream)
    assert len(text) > 0, "Expected non-empty streamed response"
    
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 3. Long streamed answer
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_long_answer(setup):
    """Unique: a very long streamed response (many chunks), producing a long-output trace."""
    if not API_KEY:
        pytest.skip("HUGGING_FACE_API_KEY not set")
    
    client = AsyncInferenceClient(api_key=API_KEY)
    
    stream = await client.chat_completion(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Write a detailed 3-paragraph essay about coffee history."},
        ],
        stream=True,
        max_tokens=500,
    )
    
    text = await _acollect_hf_stream(stream)
    assert len(text) > 200, f"Expected long response, got {len(text)} chars"
    
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 4. Multi-turn conversation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_multi_turn(setup):
    """Unique: multi-turn conversation (N inference spans), each turn streamed."""
    if not API_KEY:
        pytest.skip("HUGGING_FACE_API_KEY not set")
    
    client = AsyncInferenceClient(api_key=API_KEY)
    
    # Turn 1
    stream1 = await client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": "What is an espresso?"}],
        stream=True,
        max_tokens=100,
    )
    text1 = await _acollect_hf_stream(stream1)
    assert len(text1) > 0
    
    # Turn 2
    stream2 = await client.chat_completion(
        model=MODEL,
        messages=[
            {"role": "user", "content": "What is an espresso?"},
            {"role": "assistant", "content": text1},
            {"role": "user", "content": "And what is a latte?"},
        ],
        stream=True,
        max_tokens=100,
    )
    text2 = await _acollect_hf_stream(stream2)
    assert len(text2) > 0
    
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 5. Stream with conversation history
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_with_history(setup):
    """Unique: single streamed call primed with prior conversation history."""
    if not API_KEY:
        pytest.skip("HUGGING_FACE_API_KEY not set")
    
    client = AsyncInferenceClient(api_key=API_KEY)
    
    stream = await client.chat_completion(
        model=MODEL,
        messages=[
            {"role": "user", "content": "I love espresso."},
            {
                "role": "assistant",
                "content": "Espresso is a strong coffee shot, perfect for coffee lovers!",
            },
            {"role": "user", "content": "What about lattes?"},
        ],
        stream=True,
        max_tokens=150,
    )
    
    text = await _acollect_hf_stream(stream)
    assert len(text) > 0
    assert "latte" in text.lower() or "milk" in text.lower()
    
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 6. Single tool call
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_single_tool_call(setup):
    """Unique: stream -> one tool call -> stream final answer (2 inference spans)."""
    if not API_KEY:
        pytest.skip("HUGGING_FACE_API_KEY not set")
    
    client = AsyncInferenceClient(api_key=API_KEY)
    
    # Initial stream with tool call
    stream1 = await client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me about a latte."}],
        tools=COFFEE_TOOLS,
        stream=True,
        max_tokens=300,
    )
    
    # Consume stream and extract tool calls
    tool_calls = []
    async for chunk in stream1:
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            for tc in chunk.choices[0].delta.tool_calls:
                if tc.function:
                    tool_calls.append({
                        "id": getattr(tc, "id", f"call_{len(tool_calls)}"),
                        "name": tc.function.name,
                        "arguments": getattr(tc.function, "arguments", "{}"),
                    })
    
    if not tool_calls:
        # Model didn't call tool, just verify we have one span
        _assert_has_stream_inference(setup, min_count=1)
        return
    
    # Execute tool locally
    tool_name = tool_calls[0]["name"]
    tool_result = get_coffee_details("latte")
    
    # Stream with tool result
    stream2 = await client.chat_completion(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Tell me about a latte."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_calls[0]["id"],
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_calls[0]["arguments"],
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_calls[0]["id"],
                "content": tool_result,
            },
        ],
        tools=COFFEE_TOOLS,
        stream=True,
        max_tokens=200,
    )
    
    text2 = await _acollect_hf_stream(stream2)
    assert len(text2) > 0
    
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 7. Mixed streaming and non-streaming
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_mixed_with_non_stream(setup):
    """Unique: a non-streamed call followed by a streamed call in one trace (2 inference spans)."""
    if not API_KEY:
        pytest.skip("HUGGING_FACE_API_KEY not set")
    
    client = AsyncInferenceClient(api_key=API_KEY)
    
    # Non-streaming call
    response1 = await client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Say 'coffee' once."}],
        stream=False,
        max_tokens=50,
    )
    text1 = response1.choices[0].message.content
    assert "coffee" in text1.lower()
    
    # Streaming call
    stream2 = await client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Say 'latte' once."}],
        stream=True,
        max_tokens=50,
    )
    text2 = await _acollect_hf_stream(stream2)
    assert len(text2) > 0
    
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 8. Stream cancellation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_cancellation(setup):
    """Unique: consumer cancels/interrupts the stream mid-flight."""
    if not API_KEY:
        pytest.skip("HUGGING_FACE_API_KEY not set")
    
    client = AsyncInferenceClient(api_key=API_KEY)
    
    stream = await client.chat_completion(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Write a very long story about coffee culture."},
        ],
        stream=True,
        max_tokens=1000,
    )
    
    # Consume only a few chunks then break
    count = 0
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            count += 1
            if count >= 3:
                break  # Cancel early
    
    assert count >= 3, "Expected to consume at least 3 chunks before cancellation"
    
    # We should still have an inference span (may be incomplete)
    spans = setup.get_captured_spans()
    inference_spans = find_spans_by_type(spans, "inference") or find_spans_by_type(
        spans, "inference.framework"
    )
    assert len(inference_spans) >= 1, "Expected at least one inference span after cancellation"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
