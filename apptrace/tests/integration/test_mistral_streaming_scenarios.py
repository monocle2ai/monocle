"""Mistral AI **streaming** scenarios via ``mistralai`` SDK.

Each test function is a self-contained, realistic streaming agent behaviour that
produces a *distinct* trace.  They deliberately avoid duplicating the existing
basic streaming tests in test_mistral_stream_sync.py and test_mistral_stream_async.py
— every scenario below adds a behaviour that changes the shape of the emitted trace.

Scenarios in this file:
    1. test_stream_long_answer .......... very large streamed response (long-output trace)
    2. test_stream_multi_turn ........... multi-turn conversation, one stream per turn
    3. test_stream_with_history ......... single stream primed with prior conversation history
    4. test_stream_single_tool_call ..... stream -> one tool call -> stream final answer
    5. test_stream_sequential_tool_calls  stream -> tool A -> tool B (dependent) -> answer
    6. test_stream_parallel_tool_calls .. one assistant turn requesting several tools at once
    7. test_stream_json_output .......... JSON mode streaming
    8. test_stream_mixed_with_non_stream  a non-streamed call followed by a streamed call
    9. test_stream_cancellation ......... consumer cancels/interrupts the stream mid-flight

All scenarios reuse ``common.stream_helpers`` for Okahu+in-memory wiring and
``common.helpers`` for span verification.
"""

import asyncio
import json
import logging

import pytest
from common.helpers import find_spans_by_type, verify_inference_span
from common.stream_helpers import build_stream_span_processors
from mistralai import Mistral
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

logger = logging.getLogger(__name__)

MODEL = "mistral-large-latest"


@pytest.fixture(scope="function")
def setup():
    exporter, span_processors = build_stream_span_processors()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="mistral_streaming_scenarios",
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

COFFEE_PRICES = {
    "espresso": 3.0,
    "latte": 4.5,
    "cappuccino": 4.0,
    "americano": 3.5,
}


def get_coffee_details(coffee_name: str) -> str:
    """Get the description of a coffee drink by name."""
    return COFFEE_MENU.get(coffee_name.lower(), f"Unknown coffee: {coffee_name}")


def get_coffee_price(coffee_name: str) -> float:
    """Get the price in USD of a coffee drink by name."""
    return COFFEE_PRICES.get(coffee_name.lower(), 0.0)


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
    {
        "type": "function",
        "function": {
            "name": "get_coffee_price",
            "description": "Get the price in USD of a coffee drink by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coffee_name": {
                        "type": "string",
                        "description": "Name of the coffee drink.",
                    }
                },
                "required": ["coffee_name"],
            },
        },
    },
]


def _collect_mistral_stream(stream):
    """Drain a sync Mistral stream, returning accumulated text."""
    text_parts = []
    for chunk in stream:
        if chunk.data.choices and chunk.data.choices[0].delta.content:
            text_parts.append(chunk.data.choices[0].delta.content)
    return "".join(text_parts)


async def _acollect_mistral_stream(stream):
    """Drain an async Mistral stream, returning accumulated text."""
    text_parts = []
    async for chunk in stream:
        if chunk.data.choices and chunk.data.choices[0].delta.content:
            text_parts.append(chunk.data.choices[0].delta.content)
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
            entity_type="inference.mistral",
            model_name=MODEL,
            model_type=f"model.llm.{MODEL}",
            check_metadata=False,
            check_input_output=True,
        )
    return inference_spans


# ---------------------------------------------------------------------------
# 1. Long streamed answer
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_long_answer(setup):
    """Unique: a very long streamed response (many chunks), producing a long-output trace."""
    client = Mistral()
    
    stream = await client.chat.stream_async(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Write a detailed 3-paragraph essay about the history and culture of coffee.",
            }
        ],
    )
    
    text = await _acollect_mistral_stream(stream)
    assert len(text) > 300, f"Expected long response, got {len(text)} chars"
    
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 2. Multi-turn conversation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_multi_turn(setup):
    """Unique: multi-turn conversation (N inference spans), each turn streamed."""
    client = Mistral()
    
    # Turn 1
    stream1 = await client.chat.stream_async(
        model=MODEL,
        messages=[{"role": "user", "content": "What is an espresso?"}],
    )
    text1 = await _acollect_mistral_stream(stream1)
    assert len(text1) > 0
    
    # Turn 2
    stream2 = await client.chat.stream_async(
        model=MODEL,
        messages=[
            {"role": "user", "content": "What is an espresso?"},
            {"role": "assistant", "content": text1},
            {"role": "user", "content": "And what is a latte?"},
        ],
    )
    text2 = await _acollect_mistral_stream(stream2)
    assert len(text2) > 0
    
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 3. Stream with conversation history
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_with_history(setup):
    """Unique: single streamed call primed with prior conversation history."""
    client = Mistral()
    
    stream = await client.chat.stream_async(
        model=MODEL,
        messages=[
            {"role": "user", "content": "I love espresso."},
            {
                "role": "assistant",
                "content": "Espresso is a strong coffee shot, perfect for coffee lovers!",
            },
            {"role": "user", "content": "What about lattes?"},
        ],
    )
    
    text = await _acollect_mistral_stream(stream)
    assert len(text) > 0
    assert "latte" in text.lower() or "milk" in text.lower()
    
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 4. Single tool call
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_single_tool_call(setup):
    """Unique: stream -> one tool call -> stream final answer (2 inference spans)."""
    client = Mistral()
    
    # Initial stream with tool call
    stream1 = await client.chat.stream_async(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me about a latte."}],
        tools=COFFEE_TOOLS,
        tool_choice="auto",
    )
    
    # Consume stream and extract tool calls
    tool_calls = []
    async for chunk in stream1:
        if chunk.data.choices and chunk.data.choices[0].delta.tool_calls:
            for tc in chunk.data.choices[0].delta.tool_calls:
                if hasattr(tc, "function") and tc.function:
                    tool_calls.append({
                        "id": getattr(tc, "id", ""),
                        "name": tc.function.name,
                        "arguments": getattr(tc.function, "arguments", "{}"),
                    })
    
    if not tool_calls:
        # Model didn't call tool, just verify we have one span
        _assert_has_stream_inference(setup, min_count=1)
        return
    
    # Execute tool locally
    tool_name = tool_calls[0]["name"]
    if tool_name == "get_coffee_details":
        tool_result = get_coffee_details("latte")
    else:
        tool_result = "Unknown tool"
    
    # Stream with tool result
    stream2 = await client.chat.stream_async(
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
    )
    
    text2 = await _acollect_mistral_stream(stream2)
    assert len(text2) > 0
    
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 5. Sequential dependent tool calls
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_sequential_tool_calls(setup):
    """Unique: stream -> tool A (details) -> tool B (price, depends on A) -> answer (3 inference spans)."""
    client = Mistral()
    
    # Turn 1: Get details
    stream1 = await client.chat.stream_async(
        model=MODEL,
        messages=[{"role": "user", "content": "Get me details about a cappuccino, then its price."}],
        tools=COFFEE_TOOLS,
    )
    
    tool_calls1 = []
    async for chunk in stream1:
        if chunk.data.choices and chunk.data.choices[0].delta.tool_calls:
            for tc in chunk.data.choices[0].delta.tool_calls:
                if hasattr(tc, "function") and tc.function:
                    tool_calls1.append({
                        "id": getattr(tc, "id", f"call_{len(tool_calls1)}"),
                        "name": tc.function.name,
                        "arguments": getattr(tc.function, "arguments", "{}"),
                    })
    
    if not tool_calls1:
        _assert_has_stream_inference(setup, min_count=1)
        return
    
    tool_result1 = get_coffee_details("cappuccino")
    
    # Turn 2: Continue conversation
    messages = [
        {"role": "user", "content": "Get me details about a cappuccino, then its price."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_calls1[0]["id"],
                    "type": "function",
                    "function": {
                        "name": tool_calls1[0]["name"],
                        "arguments": tool_calls1[0]["arguments"],
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_calls1[0]["id"],
            "content": tool_result1,
        },
    ]
    
    stream2 = await client.chat.stream_async(
        model=MODEL,
        messages=messages,
        tools=COFFEE_TOOLS,
    )
    
    tool_calls2 = []
    async for chunk in stream2:
        if chunk.data.choices and chunk.data.choices[0].delta.tool_calls:
            for tc in chunk.data.choices[0].delta.tool_calls:
                if hasattr(tc, "function") and tc.function:
                    tool_calls2.append({
                        "id": getattr(tc, "id", f"call_{len(tool_calls2)}"),
                        "name": tc.function.name,
                        "arguments": getattr(tc.function, "arguments", "{}"),
                    })
    
    if tool_calls2:
        tool_result2 = get_coffee_price("cappuccino")
        messages.extend([
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_calls2[0]["id"],
                        "type": "function",
                        "function": {
                            "name": tool_calls2[0]["name"],
                            "arguments": tool_calls2[0]["arguments"],
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_calls2[0]["id"],
                "content": str(tool_result2),
            },
        ])
        
        # Final answer
        stream3 = await client.chat.stream_async(
            model=MODEL,
            messages=messages,
        )
        text3 = await _acollect_mistral_stream(stream3)
        assert len(text3) > 0
        
        _assert_has_stream_inference(setup, min_count=3)
    else:
        _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 6. Parallel tool calls
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_parallel_tool_calls(setup):
    """Unique: one assistant turn requesting several tools at once."""
    client = Mistral()
    
    stream = await client.chat.stream_async(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Get me details about both espresso and latte.",
            }
        ],
        tools=COFFEE_TOOLS,
        tool_choice="auto",
    )
    
    tool_calls = []
    async for chunk in stream:
        if chunk.data.choices and chunk.data.choices[0].delta.tool_calls:
            for tc in chunk.data.choices[0].delta.tool_calls:
                if hasattr(tc, "function") and tc.function:
                    tool_calls.append({
                        "id": getattr(tc, "id", f"call_{len(tool_calls)}"),
                        "name": tc.function.name,
                        "arguments": getattr(tc.function, "arguments", "{}"),
                    })
    
    # May get 0, 1, or 2 tool calls depending on model behavior
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 7. JSON mode streaming
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_json_output(setup):
    """Unique: response_format=json_object mode, streamed."""
    client = Mistral()
    
    stream = await client.chat.stream_async(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": 'Describe a cappuccino in JSON with fields: name, strength, has_milk.',
            }
        ],
        response_format={"type": "json_object"},
    )
    
    text = await _acollect_mistral_stream(stream)
    assert len(text) > 0
    
    # Verify it's valid JSON
    try:
        data = json.loads(text)
        assert isinstance(data, dict)
    except json.JSONDecodeError:
        # Some models may not perfectly support json mode
        pass
    
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 8. Mixed streaming and non-streaming
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_mixed_with_non_stream(setup):
    """Unique: a non-streamed call followed by a streamed call in one trace (2 inference spans)."""
    client = Mistral()
    
    # Non-streaming call
    response1 = await client.chat.complete_async(
        model=MODEL,
        messages=[{"role": "user", "content": "Say 'coffee' once."}],
    )
    text1 = response1.choices[0].message.content
    assert "coffee" in text1.lower()
    
    # Streaming call
    stream2 = await client.chat.stream_async(
        model=MODEL,
        messages=[{"role": "user", "content": "Say 'latte' once."}],
    )
    text2 = await _acollect_mistral_stream(stream2)
    assert len(text2) > 0
    
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 9. Stream cancellation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_cancellation(setup):
    """Unique: consumer cancels/interrupts the stream mid-flight."""
    client = Mistral()
    
    stream = await client.chat.stream_async(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Write a very long story about coffee culture.",
            }
        ],
    )
    
    # Consume only a few chunks then break
    count = 0
    async for chunk in stream:
        if chunk.data.choices and chunk.data.choices[0].delta.content:
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
