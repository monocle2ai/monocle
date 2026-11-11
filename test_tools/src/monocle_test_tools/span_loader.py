"""
Backward compatibility module for span loaders.

This module re-exports JSONSpanLoader and OkahuSpanLoader from their new
separate modules to maintain backward compatibility with existing code.

For new code, prefer importing directly from:
- monocle_test_tools.file_span_loader import JSONSpanLoader
- monocle_test_tools.providers.okahu.okahu_span_loader import OkahuSpanLoader
"""
from monocle_apptrace.providers.moved import moved_attr
from monocle_test_tools.file_span_loader import JSONSpanLoader

__all__ = ["JSONSpanLoader"]


def __getattr__(name):
    return moved_attr(__name__, name, {"OkahuSpanLoader": "monocle_test_tools.providers.okahu.okahu_span_loader"})
