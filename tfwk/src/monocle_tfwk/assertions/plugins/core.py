"""
Core assertion plugins for basic span operations and filtering.

This module contains the fundamental assertion plugins that provide basic
span filtering, counting, and attribute validation capabilities.
"""
from typing import TYPE_CHECKING, List

from monocle_tfwk.assertions import trace_utils
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


@plugin
class FlowAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing flow validation assertions."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "flow"
    
    def assert_flow(self, pattern: str) -> 'TraceAssertions':
        """
        Assert that the execution flow matches the given pattern using the existing FlowValidator.
        
        Pattern Syntax:
        - "->" or "→": Sequential (A must complete before B starts)
        - "||": Parallel (A and B execute with overlapping time)
        - "*": Wildcard (matches any span containing pattern)
        - "?": Optional (may or may not occur)
        - "+": One or more (repeats at least once)
        
        Examples:
        - "agent.reasoning -> tool_use -> response"
        - "preprocessing -> (embedding || retrieval) -> inference"
        - "agent.reasoning -> tool_use || knowledge_lookup"
        - "supervisor -> worker1 || worker2"
        """
        try:
            from monocle_tfwk.assertions.flow_validator import FlowValidator
            
            # Create validator directly from spans - no need for TraceGanttChart
            validator = FlowValidator.from_spans(self._current_spans)
            
            # Validate the pattern
            result = validator.validate_pattern(pattern)
            
            if not result['valid']:
                violation_details = '\n'.join([f"  - {v}" for v in result['violations']])
                suggestion_details = '\n'.join([f"  💡 {s}" for s in result['suggestions']]) if result['suggestions'] else ""
                
                error_msg = f"Flow pattern validation failed: '{pattern}'\n"
                error_msg += f"Violations:\n{violation_details}"
                if suggestion_details:
                    error_msg += f"\nSuggestions:\n{suggestion_details}"
                    
                raise AssertionError(error_msg)
                
        except ImportError as e:
            # Fallback if visualization module is not available
            raise AssertionError(f"Flow validation requires visualization module: {e}")
        except Exception as e:
            raise AssertionError(f"Flow pattern validation failed: {pattern}\nError: {str(e)}")
            
        return self
    
    def assert_conditional_flow(self, condition_agent: str, condition_output_contains: str, 
                              then_agents: List[str], else_agents: List[str] = None) -> 'TraceAssertions':
        """Assert a conditional branching flow based on agent output."""
        # Find the condition agent's output
        condition_spans = self.get_agents_by_name(condition_agent)
        assert len(condition_spans) > 0, f"Condition agent '{condition_agent}' not found"
        
        condition_output = None
        for span in condition_spans:
            output = trace_utils.get_output_from_span(span)
            if output:
                condition_output = output
                break
        
        assert condition_output is not None, f"No output found for condition agent '{condition_agent}'"
        
        # Determine which branch should be taken
        condition_met = condition_output_contains.lower() in condition_output.lower()
        
        if condition_met:
            # Verify 'then' agents were called
            for agent in then_agents:
                self.assert_agent_called(agent)
            
            # Verify 'else' agents were NOT called (if specified)
            if else_agents:
                called_agents = self.get_agent_names()
                unexpected_agents = [agent for agent in else_agents if agent in called_agents]
                assert not unexpected_agents, (
                    f"Condition was met (output contains '{condition_output_contains}') but 'else' agents were called: {unexpected_agents}"
                )
        else:
            # Verify 'else' agents were called (if specified)
            if else_agents:
                for agent in else_agents:
                    self.assert_agent_called(agent)
                
                # Verify 'then' agents were NOT called
                called_agents = self.get_agent_names()
                unexpected_agents = [agent for agent in then_agents if agent in called_agents]
                assert not unexpected_agents, (
                    f"Condition was not met (output doesn't contain '{condition_output_contains}') but 'then' agents were called: {unexpected_agents}"
                )
        
        return self