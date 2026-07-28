"""Trace-source abstraction for the Monocle test tools."""
from typing import Optional

from monocle_test_tools.trace_sources.trace_source import TraceSource
from monocle_test_tools.trace_sources.okahu_trace_source import OkahuTraceSource

__all__ = ["TraceSource", "OkahuTraceSource", "get_trace_source"]

# Known trace sources by name.
_TRACE_SOURCES = {
    OkahuTraceSource.name: OkahuTraceSource,
}


def get_trace_source(name: Optional[str]) -> Optional[TraceSource]:
    """Return a :class:`TraceSource` instance for a source name.

    Returns ``None`` for an unknown or empty name (the caller treats that as
    "no recording to do").
    """
    source_cls = _TRACE_SOURCES.get(name) if name else None
    return source_cls() if source_cls is not None else None
