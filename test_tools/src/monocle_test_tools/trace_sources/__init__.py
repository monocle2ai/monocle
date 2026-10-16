"""Trace-source abstraction for the Monocle test tools."""
from typing import Optional

from monocle_apptrace.providers import NamedRegistry
from monocle_apptrace.providers.moved import moved_attr
from monocle_test_tools.trace_sources.trace_source import TraceSource

__all__ = ["TraceSource", "get_trace_source"]

# Known trace sources by name.
_trace_sources = NamedRegistry("monocle.trace_sources", {
    "okahu": "monocle_test_tools.providers.okahu.okahu_trace_source:OkahuTraceSource",
})


def get_trace_source(name: Optional[str]) -> Optional[TraceSource]:
    """Return a :class:`TraceSource` instance for a source name.

    Returns ``None`` for an unknown or empty name (the caller treats that as
    "no recording to do").
    """
    source_cls = _trace_sources.get(name) if name else None
    return source_cls() if source_cls is not None else None


def __getattr__(name):
    return moved_attr(__name__, name, {"OkahuTraceSource": "monocle_test_tools.providers.okahu.okahu_trace_source"})
