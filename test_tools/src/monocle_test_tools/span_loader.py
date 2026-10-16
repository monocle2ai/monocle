"""
Backward compatibility module for span loaders.

For new code, import JSONSpanLoader from monocle_test_tools.file_span_loader.
"""
from monocle_apptrace.providers.moved import moved_attr
from monocle_test_tools.file_span_loader import JSONSpanLoader

__all__ = ["JSONSpanLoader"]


def __getattr__(name):
    return moved_attr(__name__, name, {"OkahuSpanLoader": "monocle_test_tools.providers.okahu.okahu_span_loader"})
