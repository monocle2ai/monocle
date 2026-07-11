"""Backward-compatibility shim.

The implementation moved to ``monocle_test_tools.span_loader.okahu_span_loader``.
This module re-exports the public names so existing imports such as
``from monocle_test_tools.okahu_span_loader import OkahuSpanLoader`` keep working.

Prefer importing from ``monocle_test_tools.span_loader`` in new code.
"""
from monocle_test_tools.span_loader.okahu_span_loader import OkahuSpanLoader

__all__ = ["OkahuSpanLoader"]
