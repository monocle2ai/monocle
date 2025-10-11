"""
Content assertion plugins for validating span inputs and outputs.

This module contains plugins that provide assertions for validating
text content in span inputs and outputs, including exact text matching
and pattern-based validation.
"""
from typing import TYPE_CHECKING

from monocle_tfwk.assertions import trace_utils
from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin

if TYPE_CHECKING:
    from monocle_tfwk.assertions.trace_assertions import TraceAssertions


@plugin
class ContentAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing content validation assertion methods."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "content"
    
    def with_input_containing(self, text: str) -> 'TraceAssertions':
        """Assert that input contains the specified text."""
        matching_spans = []
        for span in self._current_spans:
            input_text = trace_utils.get_input_from_span(span)
            if input_text and text.lower() in input_text.lower():
                matching_spans.append(span)
        assert matching_spans, f"No spans found with input containing '{text}'"
        self._current_spans = matching_spans
        return self
        
    def with_output_containing(self, text: str) -> 'TraceAssertions':
        """Assert that output contains the specified text."""
        matching_spans = []
        for span in self._current_spans:
            output_text = trace_utils.get_output_from_span(span)
            if output_text and text.lower() in output_text.lower():
                matching_spans.append(span)
        assert matching_spans, f"No spans found with output containing '{text}'"
        self._current_spans = matching_spans
        return self
    
    def contains_input(self, text: str) -> 'TraceAssertions':
        """Assert that spans contain the specified text in input."""
        return self.with_input_containing(text)
        
    def contains_output(self, text: str) -> 'TraceAssertions':
        """Assert that spans contain the specified text in output."""
        return self.with_output_containing(text)
    
    def with_input_matching_pattern(self, pattern: str) -> 'TraceAssertions':
        """Assert that input matches a regex pattern."""
        import re
        matching_spans = []
        for span in self._current_spans:
            input_text = trace_utils.get_input_from_span(span)
            if input_text and re.search(pattern, input_text, re.IGNORECASE):
                matching_spans.append(span)
        assert matching_spans, f"No spans found with input matching pattern '{pattern}'"
        self._current_spans = matching_spans
        return self
    
    def with_output_matching_pattern(self, pattern: str) -> 'TraceAssertions':
        """Assert that output matches a regex pattern."""
        import re
        matching_spans = []
        for span in self._current_spans:
            output_text = trace_utils.get_output_from_span(span)
            if output_text and re.search(pattern, output_text, re.IGNORECASE):
                matching_spans.append(span)
        assert matching_spans, f"No spans found with output matching pattern '{pattern}'"
        self._current_spans = matching_spans
        return self
    
    def with_input_not_containing(self, text: str) -> 'TraceAssertions':
        """Assert that input does NOT contain the specified text."""
        violating_spans = []
        for span in self._current_spans:
            input_text = trace_utils.get_input_from_span(span)
            if input_text and text.lower() in input_text.lower():
                violating_spans.append(span.name)
        assert not violating_spans, f"Found spans with input containing '{text}': {violating_spans}"
        return self
    
    def with_output_not_containing(self, text: str) -> 'TraceAssertions':
        """Assert that output does NOT contain the specified text."""
        violating_spans = []
        for span in self._current_spans:
            output_text = trace_utils.get_output_from_span(span)
            if output_text and text.lower() in output_text.lower():
                violating_spans.append(span.name)
        assert not violating_spans, f"Found spans with output containing '{text}': {violating_spans}"
        return self
    
    def with_input_length_between(self, min_length: int, max_length: int) -> 'TraceAssertions':
        """Assert that input text length is within specified range."""
        invalid_spans = []
        for span in self._current_spans:
            input_text = trace_utils.get_input_from_span(span)
            if input_text:
                length = len(input_text)
                if not (min_length <= length <= max_length):
                    invalid_spans.append((span.name, length))
        
        assert not invalid_spans, (
            f"Found spans with input length outside range [{min_length}, {max_length}]: {invalid_spans}"
        )
        return self
    
    def with_output_length_between(self, min_length: int, max_length: int) -> 'TraceAssertions':
        """Assert that output text length is within specified range."""
        invalid_spans = []
        for span in self._current_spans:
            output_text = trace_utils.get_output_from_span(span)
            if output_text:
                length = len(output_text)
                if not (min_length <= length <= max_length):
                    invalid_spans.append((span.name, length))
        
        assert not invalid_spans, (
            f"Found spans with output length outside range [{min_length}, {max_length}]: {invalid_spans}"
        )
        return self