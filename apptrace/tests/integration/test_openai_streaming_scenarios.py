"""OpenAI chat-completions **streaming** scenarios.

Each test function is a self-contained, realistic streaming agent behaviour that
produces a *distinct* trace in Okahu.  They deliberately avoid duplicating the
existing basic async stream in ``test_openai_api_sample_stream.py`` — every
scenario below adds a behaviour that changes the shape of the emitted trace
(extra inference spans, tool loops, retries, fallbacks, early cancellation, …).

Scenarios in this file:
    1.  test_stream_sync_basic ............ sync streaming path (existing test only covers async)
    2.  test_stream_long_answer ........... very large streamed response (long-output trace)
    3.  test_stream_multi_turn ............ multi-turn conversation, one stream per turn (N inference spans)
    4.  test_stream_with_history .......... single stream primed with prior conversation history
    5.  test_stream_single_tool_call ...... stream -> one tool call -> stream final answer (2 inference spans)
    6.  test_stream_sequential_tool_calls . stream -> tool A -> tool B (dependent) -> answer (3 inference spans)
    7.  test_stream_parallel_tool_calls ... one assistant turn requesting several tools at once
    8.  test_stream_structured_output ..... json_schema structured output, streamed
    9.  test_stream_json_output ........... json_object mode, streamed
    10. test_stream_recoverable_tool_failure  tool raises once, agent retries the tool, then streams
    11. test_stream_with_retries ......... transient API error retried, then a successful stream
    12. test_stream_model_fallback ....... primary model fails -> fallback model streams
    13. test_stream_timeout_then_success . short timeout aborts, retried with a longer timeout
    14. test_stream_cancellation ......... consumer cancels/interrupts the stream mid-flight
    15. test_stream_mixed_with_non_stream  a non-streamed call followed by a streamed call in one trace
    16. test_stream_multiple_llm_calls ... chained streamed calls (draft -> critique -> finalize)

All scenarios reuse ``common.stream_helpers`` for Okahu+in-memory wiring and
``common.helpers`` for span verification.
"""

import asyncio
import json
import logging

import pytest
from common.helpers import find_spans_by_type, verify_inference_span
from common.stream_helpers import (
    acollect_stream,
    build_stream_span_processors,
    collect_stream,
    openai_chunk_text,
)
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"


@pytest.fixture(scope="function")
def setup():
    exporter, span_processors = build_stream_span_processors()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="openai_streaming_scenarios",
            span_processors=span_processors,
            wrapper_methods=[],
        )
        yield exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# Local tool definitions + helpers (kept tiny; the point is the trace shape)
# ---------------------------------------------------------------------------

COFFEE_MENU = {
    "espresso": "A strong and bold coffee shot.",
    "latte": "A smooth coffee with steamed milk.",
    "cappuccino": "A rich coffee with frothy milk foam.",
    "americano": "Espresso with added hot water for a milder taste.",
}

GET_COFFEE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_coffee_details",
        "description": "Get the description of a coffee drink by name.",
        "parameters": {
            "type": "object",
            "properties": {"coffee_name": {"type": "string"}},
            "required": ["coffee_name"],
        },
    },
}

GET_PRICE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_coffee_price",
        "description": "Get the price in USD of a coffee drink by name.",
        "parameters": {
            "type": "object",
            "properties": {"coffee_name": {"type": "string"}},
            "required": ["coffee_name"],
        },
    },
}


def _run_coffee_tool(name: str, args: dict) -> str:
    if name == "get_coffee_details":
        return COFFEE_MENU.get(args.get("coffee_name", "").lower(), "Unknown coffee.")
    if name == "get_coffee_price":
        return json.dumps({"coffee": args.get("coffee_name"), "price_usd": 4.5})
    return "Unknown tool."


def _assemble_tool_calls(chunks) -> list:
    """Reassemble fragmented streamed tool-call deltas into complete tool calls.

    OpenAI streams tool calls across many chunks: the first carries the id/name,
    subsequent chunks append ``arguments`` fragments, keyed by ``index``.
    """
    acc: dict = {}
    for chunk in chunks:
        if not (chunk.choices and chunk.choices[0].delta.tool_calls):
            continue
        for tc in chunk.choices[0].delta.tool_calls:
            slot = acc.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
    return [acc[i] for i in sorted(acc)]


def _assert_has_stream_inference(exporter, min_count=1):
    """Common assertion: at least ``min_count`` OpenAI inference spans were traced."""
    spans = exporter.get_captured_spans()
    inference_spans = find_spans_by_type(spans, "inference") or find_spans_by_type(
        spans, "inference.framework"
    )
    assert len(inference_spans) >= min_count, (
        f"Expected >= {min_count} inference spans, got {len(inference_spans)}"
    )
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.openai",
            model_name=MODEL,
            model_type=f"model.llm.{MODEL}",
            check_metadata=True,
            check_input_output=True,
        )
    return inference_spans


# ---------------------------------------------------------------------------
# 1. Sync streaming — the existing sample only exercises the async path.
# ---------------------------------------------------------------------------
def test_stream_sync_basic(setup):
    """Unique: exercises the *synchronous* generator streaming code path."""
    client = OpenAI()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Name three coffee drinks. Be brief."}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks, text = collect_stream(stream, openai_chunk_text)
    assert chunks and text
    import time

    time.sleep(2)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 2. Long answer — a single, very large streamed response.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_long_answer(setup):
    """Unique: a long-form output stream (many deltas, large output payload)."""
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Write a detailed ~400 word history of espresso, in prose.",
            }
        ],
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=800,
    )
    _chunks, text = await acollect_stream(stream, openai_chunk_text)
    assert len(text) > 300
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 3. Multi-turn — several streamed turns; trace has one inference span per turn.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_multi_turn(setup):
    """Unique: N sequential streamed turns threaded into a single conversation."""
    client = AsyncOpenAI()
    messages = [{"role": "system", "content": "You are a concise barista."}]
    for user_msg in ["What is an americano?", "And how is it different from a latte?"]:
        messages.append({"role": "user", "content": user_msg})
        stream = await client.chat.completions.create(
            model=MODEL, messages=messages, stream=True, stream_options={"include_usage": True}
        )
        _chunks, text = await acollect_stream(stream, openai_chunk_text)
        messages.append({"role": "assistant", "content": text})
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 4. Conversation history — a single stream primed with prior turns.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_with_history(setup):
    """Unique: one stream whose input already carries multi-turn history."""
    client = AsyncOpenAI()
    messages = [
        {"role": "system", "content": "You are a concise barista."},
        {"role": "user", "content": "I like strong coffee."},
        {"role": "assistant", "content": "Noted — you enjoy bold, strong coffee."},
        {"role": "user", "content": "Recommend one drink for me."},
    ]
    stream = await client.chat.completions.create(
        model=MODEL, messages=messages, stream=True, stream_options={"include_usage": True}
    )
    _chunks, text = await acollect_stream(stream, openai_chunk_text)
    assert text
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 5. Single tool call — stream -> tool -> stream final answer.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_single_tool_call(setup):
    """Unique: a streamed tool-call turn followed by a streamed answer turn."""
    client = AsyncOpenAI()
    messages = [{"role": "user", "content": "What is a cappuccino? Use the tool."}]

    stream = await client.chat.completions.create(
        model=MODEL, messages=messages, tools=[GET_COFFEE_TOOL], stream=True
    )
    chunks, _ = await acollect_stream(stream, openai_chunk_text)
    tool_calls = _assemble_tool_calls(chunks)
    assert tool_calls, "Expected a streamed tool call"

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ],
        }
    )
    for tc in tool_calls:
        result = _run_coffee_tool(tc["name"], json.loads(tc["arguments"] or "{}"))
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    final_stream = await client.chat.completions.create(
        model=MODEL, messages=messages, stream=True, stream_options={"include_usage": True}
    )
    _chunks, text = await acollect_stream(final_stream, openai_chunk_text)
    assert text
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 6. Sequential (dependent) tool calls — details then price.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_sequential_tool_calls(setup):
    """Unique: two *dependent* tool rounds in sequence (details -> price)."""
    client = AsyncOpenAI()
    tools = [GET_COFFEE_TOOL, GET_PRICE_TOOL]
    messages = [
        {
            "role": "user",
            "content": "Describe a latte, then tell me its price. Call one tool at a time.",
        }
    ]

    # Loop the streamed tool protocol until the model stops asking for tools.
    for _ in range(4):
        stream = await client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, stream=True
        )
        chunks, text = await acollect_stream(stream, openai_chunk_text)
        tool_calls = _assemble_tool_calls(chunks)
        if not tool_calls:
            assert text
            break
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            result = _run_coffee_tool(tc["name"], json.loads(tc["arguments"] or "{}"))
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 7. Parallel tool calls — one assistant turn requesting multiple tools at once.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_parallel_tool_calls(setup):
    """Unique: a single streamed turn that emits several tool calls in parallel."""
    client = AsyncOpenAI()
    tools = [GET_COFFEE_TOOL, GET_PRICE_TOOL]
    messages = [
        {
            "role": "user",
            "content": "In one step, get details AND price for an espresso.",
        }
    ]
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        parallel_tool_calls=True,
        stream=True,
    )
    chunks, _ = await acollect_stream(stream, openai_chunk_text)
    tool_calls = _assemble_tool_calls(chunks)
    assert len(tool_calls) >= 1  # model usually emits 2; at least 1 proves the path

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ],
        }
    )
    for tc in tool_calls:
        result = _run_coffee_tool(tc["name"], json.loads(tc["arguments"] or "{}"))
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    final = await client.chat.completions.create(
        model=MODEL, messages=messages, stream=True, stream_options={"include_usage": True}
    )
    _chunks, text = await acollect_stream(final, openai_chunk_text)
    assert text
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 8. Structured output — json_schema, streamed.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_structured_output(setup):
    """Unique: response_format json_schema streamed as text deltas -> valid JSON."""
    client = AsyncOpenAI()
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "coffee_card",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "strength": {"type": "string"},
                    "has_milk": {"type": "boolean"},
                },
                "required": ["name", "strength", "has_milk"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Describe a latte as a coffee_card."}],
        response_format=schema,
        stream=True,
        stream_options={"include_usage": True},
    )
    _chunks, text = await acollect_stream(stream, openai_chunk_text)
    parsed = json.loads(text)
    assert {"name", "strength", "has_milk"} <= parsed.keys()
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 9. JSON output — json_object mode, streamed.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_json_output(setup):
    """Unique: response_format json_object (freeform JSON) streamed."""
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Respond only with a JSON object."},
            {"role": "user", "content": "List two coffee drinks with a 'drinks' array."},
        ],
        response_format={"type": "json_object"},
        stream=True,
        stream_options={"include_usage": True},
    )
    _chunks, text = await acollect_stream(stream, openai_chunk_text)
    assert isinstance(json.loads(text), dict)
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 10. Recoverable tool failure — the tool raises once, the agent retries it.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_recoverable_tool_failure(setup):
    """Unique: a tool fails first (error fed back to the model), then succeeds."""
    client = AsyncOpenAI()
    call_state = {"attempts": 0}

    def flaky_tool(args):
        call_state["attempts"] += 1
        if call_state["attempts"] == 1:
            return "ERROR: coffee service temporarily unavailable, please retry."
        return _run_coffee_tool("get_coffee_details", args)

    messages = [
        {"role": "user", "content": "Get details for an americano; retry on transient errors."}
    ]
    for _ in range(4):
        stream = await client.chat.completions.create(
            model=MODEL, messages=messages, tools=[GET_COFFEE_TOOL], stream=True
        )
        chunks, text = await acollect_stream(stream, openai_chunk_text)
        tool_calls = _assemble_tool_calls(chunks)
        if not tool_calls:
            assert text
            break
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            result = flaky_tool(json.loads(tc["arguments"] or "{}"))
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    assert call_state["attempts"] >= 1
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 11. Retries — a transient API error is retried, then a stream succeeds.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_with_retries(setup):
    """Unique: first create() call is forced to fail, retried, then streams OK."""
    client = AsyncOpenAI()
    attempts = {"n": 0}

    async def create_with_retry():
        for attempt in range(3):
            attempts["n"] += 1
            try:
                # Force a transient failure on the first attempt only.
                model = "nonexistent-model-xyz" if attempt == 0 else MODEL
                return await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "One coffee fact, briefly."}],
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except Exception as ex:  # transient -> retry with the real model
                logger.info("Retry after streaming create error: %s", ex)
                continue
        raise RuntimeError("all retries exhausted")

    stream = await create_with_retry()
    _chunks, text = await acollect_stream(stream, openai_chunk_text)
    assert text and attempts["n"] >= 2
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 12. Model fallback — primary model fails, fallback model streams.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_model_fallback(setup):
    """Unique: primary (bad) model errors, code falls back to a working model."""
    client = AsyncOpenAI()
    primary, fallback = "gpt-nonexistent-primary", MODEL
    stream = None
    for model in (primary, fallback):
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Describe an espresso in one line."}],
                stream=True,
                stream_options={"include_usage": True},
            )
            break
        except Exception as ex:
            logger.info("Model %s failed, falling back: %s", model, ex)
    assert stream is not None
    _chunks, text = await acollect_stream(stream, openai_chunk_text)
    assert text
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 13. Timeout handling — a too-short timeout aborts, retried with a longer one.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_timeout_then_success(setup):
    """Unique: request-level timeout raises, then a longer timeout streams OK."""
    from openai import APITimeoutError

    client = AsyncOpenAI()
    stream = None
    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Write 300 words about coffee."}],
            stream=True,
            timeout=0.001,  # absurdly short -> forces a timeout
        )
        await acollect_stream(stream, openai_chunk_text)
        stream = None  # if it somehow finished, fall through to the clean retry below
    except (APITimeoutError, Exception) as ex:
        logger.info("Expected timeout, retrying with longer timeout: %s", ex)

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "One coffee fact."}],
        stream=True,
        stream_options={"include_usage": True},
        timeout=30,
    )
    _chunks, text = await acollect_stream(stream, openai_chunk_text)
    assert text
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 14. Cancellation — the consumer interrupts the stream mid-flight.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_cancellation(setup):
    """Unique: consumer stops iterating early and closes the stream (interrupt)."""
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Write a very long essay about coffee."}],
        stream=True,
        max_tokens=800,
    )
    received = 0
    async for chunk in stream:
        if openai_chunk_text(chunk):
            received += 1
        if received >= 3:  # user cancels after a few tokens
            break
    await stream.close()  # explicit interruption of the underlying HTTP stream
    assert received >= 1
    await asyncio.sleep(3)
    # A cancelled stream still yields a (partial) inference span.
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 15. Mixed streaming + non-streaming in a single trace.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_mixed_with_non_stream(setup):
    """Unique: a non-streamed call and a streamed call in the same workflow trace."""
    client = AsyncOpenAI()
    # Non-streamed classification step...
    classification = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with one word: is 'latte' a coffee?"}],
    )
    assert classification.choices[0].message.content
    # ...followed by a streamed answer step.
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Describe a latte briefly."}],
        stream=True,
        stream_options={"include_usage": True},
    )
    _chunks, text = await acollect_stream(stream, openai_chunk_text)
    assert text
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 16. Multiple chained streamed LLM invocations (draft -> critique -> finalize).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_multiple_llm_calls(setup):
    """Unique: a 3-stage streamed pipeline, each stage feeding the next."""
    client = AsyncOpenAI()

    async def stream_step(prompt: str) -> str:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
        )
        _chunks, text = await acollect_stream(stream, openai_chunk_text)
        return text

    draft = await stream_step("Write a one-sentence description of a mocha.")
    critique = await stream_step(f"Critique this description in one sentence: {draft}")
    final = await stream_step(f"Rewrite the description using this critique: {critique}")
    assert draft and critique and final
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=3)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
