"""
Backward compatibility module for span loaders.

This module re-exports JSONSpanLoader and OkahuSpanLoader from their new
separate modules to maintain backward compatibility with existing code.

For new code, prefer importing directly from:
- monocle_test_tools.file_span_loader import JSONSpanLoader
- monocle_test_tools.okahu_span_loader import OkahuSpanLoader
"""
from monocle_test_tools.file_span_loader import JSONSpanLoader
from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

__all__ = ["JSONSpanLoader", "OkahuSpanLoader"]
