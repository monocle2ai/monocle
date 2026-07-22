"""OpenAI **Responses API** streaming scenarios (event-based SSE path).

The Responses API streams *typed events* (``response.output_text.delta``,
``response.completed`` …) rather than the ``chat.completion.chunk`` objects used
by the chat-completions API.  This exercises the ``handle_event`` branch of the
OpenAI stream processor — a different code path from every chat-completions
scenario.

The existing ``test_openai_response_sample.py`` already covers a *basic* async
Responses stream, so this file only adds shapes that produce a different trace:

    1. test_responses_stream_structured_output . streamed json_schema structured output
    2. test_responses_stream_multi_turn_chained  multi-turn via previous_response_id (server-side history)
"""

import asyncio
import json
import logging

import pytest
from common.helpers import find_spans_by_type, verify_inference_span
from common.stream_helpers import build_stream_span_processors
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"


@pytest.fixture(scope="function")
def setup():
    exporter, span_processors = build_stream_span_processors()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="openai_responses_streaming_scenarios",
            span_processors=span_processors,
            wrapper_methods=[],
        )
        yield exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


async def _drain_responses_stream(stream):
    """Collect text deltas and the final response id from a Responses stream."""
    text_parts, response_id = [], None
    async for event in stream:
        etype = getattr(event, "type", "")
        if etype == "response.output_text.delta":
            text_parts.append(event.delta)
        elif etype == "response.completed":
            response_id = getattr(event.response, "id", None)
    return "".join(text_parts), response_id


def _assert_has_stream_inference(exporter, min_count=1):
    spans = exporter.get_captured_spans()
    inference_spans = find_spans_by_type(spans, "inference") or find_spans_by_type(
        spans, "inference.framework"
    )
    assert len(inference_spans) >= min_count
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
# 1. Structured output via the Responses API, streamed.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_responses_stream_structured_output(setup):
    """Unique: json_schema structured output over the event-based Responses stream."""
    client = AsyncOpenAI()
    stream = await client.responses.create(
        model=MODEL,
        input="Describe a cappuccino.",
        text={
            "format": {
                "type": "json_schema",
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
            }
        },
        stream=True,
    )
    text, _rid = await _drain_responses_stream(stream)
    parsed = json.loads(text)
    assert {"name", "strength", "has_milk"} <= parsed.keys()
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=1)


# ---------------------------------------------------------------------------
# 2. Multi-turn Responses streaming chained via previous_response_id.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_responses_stream_multi_turn_chained(setup):
    """Unique: server-side conversation state via previous_response_id across streamed turns."""
    client = AsyncOpenAI()

    first = await client.responses.create(
        model=MODEL, input="My favorite coffee is espresso. Acknowledge briefly.", stream=True
    )
    text1, rid1 = await _drain_responses_stream(first)
    assert text1 and rid1, "Expected first response text and id"

    # Second turn references the first purely via previous_response_id (no re-sent history).
    second = await client.responses.create(
        model=MODEL,
        previous_response_id=rid1,
        input="What did I say my favorite coffee was?",
        stream=True,
    )
    text2, _rid2 = await _drain_responses_stream(second)
    assert text2
    await asyncio.sleep(3)
    _assert_has_stream_inference(setup, min_count=2)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
