"""Backward-compatibility shim.

The implementation moved to ``monocle_test_tools.span_loader.file_span_loader``.
This module re-exports the public names so existing imports such as
``from monocle_test_tools.file_span_loader import JSONSpanLoader`` keep working.

Prefer importing from ``monocle_test_tools.span_loader`` in new code.
"""
from monocle_test_tools.span_loader.file_span_loader import JSONSpanLoader

__all__ = ["JSONSpanLoader"]
