"""
Core assertion plugins for basic span operations and filtering.

This module contains the fundamental assertion plugins that provide basic
span filtering, counting, and attribute validation capabilities.
"""
from typing import TYPE_CHECKING

from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin

if TYPE_CHECKING:
    from monocle_tfwk.assertions.trace_assertions import TraceAssertions


@plugin
class CoreSpanAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing core span assertion methods."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "core_spans"
    
    def filter_by_name(self, name: str) -> 'TraceAssertions':
        """Filter spans by name."""
        matching_spans = [
            span for span in self._current_spans
            if span.name == name
        ]
        self._current_spans = matching_spans
        return self
        
    def filter_by_attribute(self, key: str, value: str = None) -> 'TraceAssertions':
        """Filter spans by attribute key and optionally value."""
        matching_spans = []
        for span in self._current_spans:
            if span.attributes and key in span.attributes:
                if value is None or span.attributes.get(key) == value:
                    matching_spans.append(span)
        self._current_spans = matching_spans
        return self
    
    def exactly(self, count: int) -> 'TraceAssertions':
        """Assert exact count of matching spans."""
        actual_count = len(self._current_spans)
        assert actual_count == count, f"Expected {count} spans, found {actual_count}"
        return self
        
    def at_least(self, count: int) -> 'TraceAssertions':
        """Assert minimum count of matching spans."""
        actual_count = len(self._current_spans)
        assert actual_count >= count, f"Expected at least {count} spans, found {actual_count}"
        return self
        
    def assert_spans(self, min_count: int = None, max_count: int = None, count: int = None) -> 'TraceAssertions':
        """Assert span count conditions."""
        actual_count = len(self._current_spans)
        
        if count is not None:
            assert actual_count == count, f"Expected exactly {count} spans, found {actual_count}"
        elif min_count is not None:
            assert actual_count >= min_count, f"Expected at least {min_count} spans, found {actual_count}"
        elif max_count is not None:
            assert actual_count <= max_count, f"Expected at most {max_count} spans, found {actual_count}"
            
        return self
        
    def assert_span_with_name(self, name: str) -> 'TraceAssertions':
        """Assert that at least one span exists with the given name."""
        matching_spans = [span for span in self._current_spans if span.name == name]
        assert matching_spans, f"No span found with name '{name}'"
        return self
        
    def assert_attribute(self, key: str, value: str = None) -> 'TraceAssertions':
        """Assert that spans have the specified attribute, optionally with a specific value."""
        matching_spans = []
        for span in self._current_spans:
            if span.attributes and key in span.attributes:
                if value is None or span.attributes.get(key) == value:
                    matching_spans.append(span)
        
        if value is not None:
            assert matching_spans, f"No spans found with attribute '{key}' = '{value}'"
        else:
            assert matching_spans, f"No spans found with attribute '{key}'"
            
        self._current_spans = matching_spans
        return self
    
    def completed_successfully(self) -> 'TraceAssertions':
        """Assert that spans completed without errors."""
        error_spans = [
            span for span in self._current_spans
            if span.status.status_code.name == "ERROR"
        ]
        assert not error_spans, f"Found {len(error_spans)} spans with errors"
        return self


@plugin
class QueryAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing JMESPath query and entity-based assertions."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "query"
    
    def assert_entity_type(self, entity_type: str) -> 'TraceAssertions':
        """Assert that traces contain a specific entity type."""
        entities = self.query_engine.find_entities_by_type(entity_type)
        assert len(entities) > 0, f"Entity type '{entity_type}' not found in traces"
        return self
    
    def assert_workflow_complete(self, expected_agent_type: str = "agent.openai_agents") -> 'TraceAssertions':
        """Assert that we have a complete agent workflow."""
        complete = self.query_engine.assert_agent_workflow(expected_agent_type)
        assert complete, f"Incomplete agent workflow for type '{expected_agent_type}'"
        return self


@plugin 
class PerformanceAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing performance and timing assertions."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "performance"
    
    def within_time_limit(self, max_seconds: float) -> 'TraceAssertions':
        """Assert that operations completed within time limit."""
        slow_spans = []
        for span in self._current_spans:
            duration = (span.end_time - span.start_time) / 1_000_000_000  # Convert to seconds
            if duration > max_seconds:
                slow_spans.append((span, duration))
        assert not slow_spans, f"Found spans exceeding {max_seconds}s: {[(s.name, d) for s, d in slow_spans]}"
        return self
    
    def assert_total_duration_under(self, max_seconds: float) -> 'TraceAssertions':
        """Assert that the total duration of all current spans is under the specified limit."""
        total_duration = 0.0
        for span in self._current_spans:
            duration = (span.end_time - span.start_time) / 1_000_000_000  # Convert to seconds
            total_duration += duration
        
        assert total_duration <= max_seconds, (
            f"Total duration {total_duration:.3f}s exceeds limit of {max_seconds}s"
        )
        return self
    
    def assert_average_duration_under(self, max_seconds: float) -> 'TraceAssertions':
        """Assert that the average duration of current spans is under the specified limit."""
        if not self._current_spans:
            return self
            
        total_duration = 0.0
        for span in self._current_spans:
            duration = (span.end_time - span.start_time) / 1_000_000_000  # Convert to seconds
            total_duration += duration
        
        average_duration = total_duration / len(self._current_spans)
        
        assert average_duration <= max_seconds, (
            f"Average duration {average_duration:.3f}s exceeds limit of {max_seconds}s"
        )
        return self