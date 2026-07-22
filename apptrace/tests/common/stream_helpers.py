"""Shared helpers for streaming scenario examples.

These utilities are reused by every ``test_*_streaming_scenarios.py`` example so
that each scenario stays small and focused on the *agent behaviour* being traced
rather than on telemetry plumbing.

Two things are centralised here:

1. ``build_stream_span_processors`` — wires an in-memory/console exporter (for
   local assertions) together with the Okahu exporter (so every run also
   publishes a real trace to Okahu).  The Okahu exporter is always added;
   ``OkahuSpanExporter`` raises if ``OKAHU_API_KEY`` is not resolvable, so a
   misconfigured environment fails loudly instead of silently running offline.
2. ``collect_stream`` / ``acollect_stream`` — drain a streamed response while
   accumulating the text deltas, mirroring how a real UI would consume a stream.
"""

from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.okahu.okahu_exporter import OkahuSpanExporter


def build_stream_span_processors(exporter=None):
    """Return span processors that export to BOTH an in-memory exporter and Okahu.

    The in-memory ``CustomConsoleSpanExporter`` is what each scenario asserts on
    locally (via ``get_captured_spans``).  The ``OkahuSpanExporter`` publishes the
    same trace to Okahu so the scenario shows up in the Okahu dashboard.  We use
    ``SimpleSpanProcessor`` for both so every span is delivered synchronously and
    no explicit flush is required before the process exits.

    Returns ``(exporter, span_processors)``.
    """
    exporter = exporter or CustomConsoleSpanExporter()
    span_processors = [
        SimpleSpanProcessor(exporter),
        SimpleSpanProcessor(OkahuSpanExporter()),
    ]
    return exporter, span_processors


def collect_stream(stream, text_fn):
    """Drain a synchronous stream, returning (chunks, concatenated_text).

    ``text_fn(chunk)`` extracts the incremental text from a chunk (or ``None``).
    """
    chunks, parts = [], []
    for chunk in stream:
        chunks.append(chunk)
        piece = text_fn(chunk)
        if piece:
            parts.append(piece)
    return chunks, "".join(parts)


async def acollect_stream(stream, text_fn):
    """Async counterpart of :func:`collect_stream`."""
    chunks, parts = [], []
    async for chunk in stream:
        chunks.append(chunk)
        piece = text_fn(chunk)
        if piece:
            parts.append(piece)
    return chunks, "".join(parts)


def openai_chunk_text(chunk):
    """Extract the text delta from an OpenAI ``chat.completion.chunk``."""
    if getattr(chunk, "choices", None) and chunk.choices[0].delta.content:
        return chunk.choices[0].delta.content
    return None
