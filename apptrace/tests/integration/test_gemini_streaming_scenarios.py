"""Google **Gemini** streaming scenarios (via ``google.genai``).

Each test function is a self-contained, realistic streaming behaviour that
produces a *distinct* trace in Okahu.  They deliberately avoid duplicating the
existing basic streams in ``test_gemini_stream.py`` — that file already covers:

    * a basic ``client.models.generate_content_stream`` call, and
    * a basic multi-turn ``chat.send_message_stream`` conversation.

Every scenario below adds a behaviour that changes the *shape* of the emitted
trace (function-call parts, structured JSON output, long output, or mixed
streamed + non-streamed calls in one trace).

Scenarios in this file:
    1. test_stream_function_calling ...... stream a tool (function-declaration) turn,
                                           run a local Python function, then stream a
                                           follow-up turn with the tool result
                                           (>=2 inference spans, FUNCTION_CALL path).
    2. test_stream_structured_output ..... response_mime_type=application/json +
                                           response_schema, streamed -> valid JSON.
    3. test_stream_long_answer ........... one long streamed response (many chunks),
                                           concatenated text asserted to be long.
    4. test_stream_mixed_with_non_stream . a non-streamed generate_content followed by a
                                           streamed generate_content_stream in one trace
                                           (>=2 inference spans). Chosen over the
                                           system-instruction/tool-follow-up variant
                                           because it produces a cleanly distinct
                                           mixed-mode trace with the installed API.

All scenarios reuse ``common.stream_helpers.build_stream_span_processors`` for
Okahu + in-memory wiring, and ``common.helpers`` for span verification.

Client construction, model name, and API usage match ``test_gemini_stream.py``
exactly: ``google.genai.Client(api_key=os.getenv("GEMINI_API_KEY"))`` and model
``gemini-2.5-flash``.  The Gemini stream processor extracts ``function_call``
parts and sets finish_reason FUNCTION_CALL; inference spans carry
``entity.1.type == "inference.gemini"`` (confirmed in ``test_gemini_stream.py``'s
``verify_inference_span(..., entity_type="inference.gemini", ...)``).
"""

import json
import logging
import os
import time

import pytest
from common.helpers import find_spans_by_type, verify_inference_span
from common.stream_helpers import build_stream_span_processors
from google import genai
from google.genai import types
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"


@pytest.fixture(scope="function")
def setup():
    exporter, span_processors = build_stream_span_processors()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="gemini_streaming_scenarios",
            span_processors=span_processors,
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

# A Gemini function declaration wrapped in a Tool. ``parameters`` is a
# ``types.Schema`` describing the JSON arguments the model should produce.
GET_COFFEE_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_coffee_details",
            description="Get the description of a coffee drink by name.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "coffee_name": types.Schema(
                        type=types.Type.STRING,
                        description="Name of the coffee drink, e.g. 'latte'.",
                    ),
                },
                required=["coffee_name"],
            ),
        )
    ]
)


def _run_coffee_tool(name: str, args: dict) -> str:
    """Local implementation the agent calls after the model requests the tool."""
    if name == "get_coffee_details":
        return COFFEE_MENU.get(
            str(args.get("coffee_name", "")).lower(), "Unknown coffee."
        )
    return "Unknown tool."


def collect_gemini_stream(stream):
    """Drain a Gemini stream, returning (chunks, text, function_calls).

    Mirrors how a real UI consumes a stream: accumulate text deltas and collect
    any ``function_call`` parts the model emits (finish_reason FUNCTION_CALL).
    """
    chunks, parts, function_calls = [], [], []
    for chunk in stream:
        chunks.append(chunk)
        if chunk.text:
            parts.append(chunk.text)
        if chunk.function_calls:
            function_calls.extend(chunk.function_calls)
    return chunks, "".join(parts), function_calls


def _assert_has_stream_inference(exporter, min_count=1):
    """Common assertion: at least ``min_count`` Gemini inference spans were traced.

    Uses the same ``entity_type="inference.gemini"`` that ``test_gemini_stream.py``
    asserts on.  ``check_input_output`` is left on so we confirm data.input /
    data.output events exist, matching the existing gemini test's verification.
    """
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
            entity_type="inference.gemini",
            model_name=MODEL,
            model_type=f"model.llm.{MODEL}",
            check_metadata=True,
            check_input_output=True,
        )
    return inference_spans


# ---------------------------------------------------------------------------
# 1. Function calling / tool streaming.
#    Unique: a streamed turn that emits a ``function_call`` part (FUNCTION_CALL
#    finish reason), a local Python function is run, and the tool result is fed
#    back in a second streamed turn -> two inference spans, tool-loop trace.
# ---------------------------------------------------------------------------
def test_stream_function_calling(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="What is a cappuccino? Use the get_coffee_details tool."
                )
            ],
        )
    ]
    tool_config = types.GenerateContentConfig(tools=[GET_COFFEE_TOOL])

    stream = client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=tool_config,
    )
    _chunks, _text, function_calls = collect_gemini_stream(stream)
    assert function_calls, "Expected a streamed function_call part"

    # Assemble the streamed function call and run the local tool.
    fc = function_calls[0]
    tool_result = _run_coffee_tool(fc.name, dict(fc.args or {}))

    # Feed the model's function_call turn + our function_response back in, and
    # stream the follow-up answer.
    contents.append(
        types.Content(
            role="model",
            parts=[types.Part.from_function_call(name=fc.name, args=dict(fc.args or {}))],
        )
    )
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": tool_result},
                )
            ],
        )
    )

    follow_up = client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=tool_config,
    )
    _chunks2, text2, _fc2 = collect_gemini_stream(follow_up)
    assert text2, "Expected a streamed answer after the tool result"

    time.sleep(3)
    _assert_has_stream_inference(setup, min_count=2)


# ---------------------------------------------------------------------------
# 2. Structured output (JSON schema).
#    Unique: response_mime_type="application/json" + response_schema forces the
#    streamed text deltas to concatenate into a single valid JSON object.
# ---------------------------------------------------------------------------
def test_stream_structured_output(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # ``response_schema`` accepts a types.Schema in google-genai 1.75.0.
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "name": types.Schema(type=types.Type.STRING),
            "strength": types.Schema(type=types.Type.STRING),
            "has_milk": types.Schema(type=types.Type.BOOLEAN),
        },
        required=["name", "strength", "has_milk"],
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )

    stream = client.models.generate_content_stream(
        model=MODEL,
        contents="Describe a latte with fields name, strength, has_milk.",
        config=config,
    )
    _chunks, text, _fc = collect_gemini_stream(stream)
    parsed = json.loads(text)
    assert {"name", "strength", "has_milk"} <= parsed.keys()

    time.sleep(3)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 3. Long-answer streaming.
#    Unique: a single long-form output stream (many text deltas, large payload).
# ---------------------------------------------------------------------------
def test_stream_long_answer(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    config = types.GenerateContentConfig(max_output_tokens=2000)
    stream = client.models.generate_content_stream(
        model=MODEL,
        contents="Write a detailed ~400 word history of espresso, in flowing prose.",
        config=config,
    )
    chunks, text, _fc = collect_gemini_stream(stream)
    assert len(chunks) > 1, "Expected the answer to arrive across many chunks"
    assert len(text) > 300, "Expected a long concatenated response"

    time.sleep(3)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 4. Mixed streaming + non-streaming in one trace.
#    Unique: a non-streamed ``generate_content`` classification step followed by
#    a streamed ``generate_content_stream`` answer step in the same workflow ->
#    two inference spans of different (streamed vs non-streamed) shape.
#
#    Chosen over the "system-instruction + multi-turn tool follow-up" variant:
#    the mixed-mode trace is cleanly distinct from scenario 1 (which already
#    covers tools/system-style guidance) and exercises both the streamed and
#    non-streamed instrumentation paths in a single trace.
# ---------------------------------------------------------------------------
def test_stream_mixed_with_non_stream(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Non-streamed step.
    classification = client.models.generate_content(
        model=MODEL,
        contents="Reply with exactly one word: is 'latte' a coffee?",
    )
    assert classification.text, "Expected a non-streamed classification response"

    # Streamed step in the same trace/workflow.
    stream = client.models.generate_content_stream(
        model=MODEL,
        contents="Describe a latte briefly.",
    )
    _chunks, text, _fc = collect_gemini_stream(stream)
    assert text, "Expected a streamed answer"

    time.sleep(3)
    _assert_has_stream_inference(setup, min_count=2)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
