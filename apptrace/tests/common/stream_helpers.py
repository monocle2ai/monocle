"""Shared helpers for streaming scenario examples.

These utilities are reused by every ``test_*_streaming_scenarios.py`` example so
that each scenario stays small and focused on the *agent behaviour* being traced
rather than on telemetry plumbing.

Two things are centralised here:

1. ``build_stream_span_processors`` — wires an in-memory/console exporter (for
   local assertions) together with the Okahu exporter (so every run also
   publishes a real trace to Okahu).  The Okahu exporter is added only when an
   ``OKAHU_API_KEY`` is resolvable, so the examples still run offline.
2. ``collect_stream`` / ``acollect_stream`` — drain a streamed response while
   accumulating the text deltas, mirroring how a real UI would consume a stream.
"""

import logging
import os
from pathlib import Path

from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from common.custom_exporter import CustomConsoleSpanExporter

logger = logging.getLogger(__name__)


def _ensure_okahu_key_loaded() -> bool:
    """Best-effort load of ``OKAHU_API_KEY`` from the repo-root ``.env.monocle``.

    ``get_monocle_env_value`` only looks at ``os.getcwd()/.env.monocle``; pytest
    may run from a different working directory, so we walk up from this file to
    find the repo root and hoist the key into ``os.environ``.
    """
    if os.environ.get("OKAHU_API_KEY"):
        return True
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env.monocle"
        if not env_file.exists():
            continue
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OKAHU_API_KEY="):
                    os.environ["OKAHU_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
                    return True
        except Exception:  # pragma: no cover - defensive, never break a scenario
            pass
    return bool(os.environ.get("OKAHU_API_KEY"))


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
    span_processors = [SimpleSpanProcessor(exporter)]

    if _ensure_okahu_key_loaded():
        try:
            from monocle_apptrace.exporters.okahu.okahu_exporter import OkahuSpanExporter

            span_processors.append(SimpleSpanProcessor(OkahuSpanExporter()))
            logger.info("Okahu exporter enabled for streaming scenario.")
        except Exception as ex:  # missing key / import issue -> keep running offline
            logger.warning("Okahu exporter not enabled (%s); running with in-memory only.", ex)
    else:
        logger.info("OKAHU_API_KEY not found; streaming scenario runs in-memory only.")

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
