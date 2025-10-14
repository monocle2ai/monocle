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


@plugin
class FlowAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing flow validation assertions."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "flow"
    
    def assert_flow(self, pattern: str, parent_filter: str = None) -> 'TraceAssertions':
        """
        Assert that spans follow a specific flow pattern within common parent contexts.
        
        Validates that spans under the same parent follow the specified pattern.
        This focuses on parent-child relationships rather than global timing.
        
        Args:
            pattern: Flow pattern using -> (sequence), || (parallel), and grouping
            parent_filter: Optional pattern to filter which parents to validate under
        
        Flow Pattern Syntax:
        - "A -> B -> C" - Sequential execution within parent scope
        - "A || B" - Parallel execution within parent scope
        - "(A || B) -> C" - Mixed patterns within parent scope
        - Supports wildcards: *, ?
        - Field prefixes: "name:pattern", "type:pattern"
        
        Examples:
        - "agent.reasoning -> tool_use -> response"
        - "preprocessing -> (embedding || retrieval) -> inference"
        - "agent.reasoning -> tool_use || knowledge_lookup", parent_filter="workflow"
        - "worker1 || worker2", parent_filter="type:supervisor"
        """
        
        def group_spans_by_parent(spans) -> dict:
            """Group spans by their parent ID."""
            groups = {}
            
            for span in spans:
                # Extract parent ID from span.parent SpanContext
                parent_id = None
                if hasattr(span, 'parent') and span.parent:
                    parent_id = hex(span.parent.span_id) if hasattr(span.parent, 'span_id') else None
                
                if parent_id not in groups:
                    groups[parent_id] = []
                groups[parent_id].append(span)
                
            return groups
        
        def find_parent_span(parent_id: str, all_spans):
            """Find parent span by ID."""
            for span in all_spans:
                # Extract span ID from span.context
                span_id = None
                if hasattr(span, 'context') and span.context:
                    span_id = hex(span.context.span_id) if hasattr(span.context, 'span_id') else None
                if span_id == parent_id:
                    return span
            return None
        
        def get_parent_info(parent_id: str, all_spans) -> str:
            """Get human-readable parent information for error messages."""
            if not parent_id:
                return " (root level)"
            
            parent_span = find_parent_span(parent_id, all_spans)
            if parent_span:
                parent_type = parent_span.attributes.get('span.type', 'unknown') if parent_span.attributes else 'unknown'
                return f" (under parent: {parent_span.name} [{parent_type}])"
            else:
                return f" (under parent ID: {parent_id})"
        
        def match_field_pattern(field_value: str, pattern: str) -> bool:
            """Match a specific field value against a pattern."""
            import re
            
            if '*' in pattern or '?' in pattern:
                regex_pattern = pattern.replace('*', '.*').replace('?', '.?')
                return re.match(regex_pattern, field_value, re.IGNORECASE) is not None
            
            return (field_value == pattern or pattern.lower() in field_value.lower())
        
        def matches_pattern(span, pattern: str) -> bool:
            """
            Check if span matches a pattern (supports wildcards).
            
            Pattern matching order:
            1. span.name (exact match)
            2. span.type (exact match) 
            3. span.name (partial match for non-wildcard patterns)
            4. Regex match for wildcard patterns (* and ?)
            
            Use prefixes for specific field matching:
            - "name:pattern" - match only span.name
            - "type:pattern" - match only span.type
            """
            import re
            
            # Handle explicit field prefixes
            if ':' in pattern:
                field, value = pattern.split(':', 1)
                if field == 'name':
                    return match_field_pattern(span.name, value)
                elif field == 'type':
                    span_type = span.attributes.get('span.type', '') if span.attributes else ''
                    return match_field_pattern(span_type, value)
            
            # Convert wildcard pattern to regex
            if '*' in pattern or '?' in pattern:
                regex_pattern = pattern.replace('*', '.*').replace('?', '.?')
                span_type = span.attributes.get('span.type', '') if span.attributes else ''
                return (re.match(regex_pattern, span.name, re.IGNORECASE) or
                        re.match(regex_pattern, span_type, re.IGNORECASE))
            
            # Exact match (name, type, then partial name)
            span_type = span.attributes.get('span.type', '') if span.attributes else ''
            return (span.name == pattern or 
                    span_type == pattern or
                    pattern.lower() in span.name.lower())
        
        def should_process_group(parent_id: str) -> bool:
            """Check if a parent group should be processed based on parent_filter."""
            if not parent_filter:
                return True  # Process all groups if no filter
            
            if not parent_id:
                return False  # Skip root spans if parent filter is specified
            
            parent_span = find_parent_span(parent_id, self._current_spans)
            return parent_span and matches_pattern(parent_span, parent_filter)
        
        def format_validation_error(all_errors: list, single_group: bool = False) -> str:
            """Format validation error message using consistent multi-group style."""
            # Use consistent multi-group format for both single and multiple errors
            error_msg = f"Flow pattern validation failed for '{pattern}' - no matching groups found.\n"
            error_msg += f"Checked {len(all_errors)} parent group{'s' if len(all_errors) != 1 else ''}:\n"
            
            # Show up to 3 groups with detailed violations
            for i, error_info in enumerate(all_errors[:3]):
                error_msg += f"\n{i+1}. Group{error_info['parent_info']}:\n"
                error_msg += f"   Violations: {', '.join(error_info['violations'])}\n"
            
            if len(all_errors) > 3:
                error_msg += f"   ... and {len(all_errors) - 3} more groups\n"
            
            return error_msg

        try:
            from monocle_tfwk.assertions.flow_validator import FlowValidator
            
            # Group spans by common parent and validate pattern within each group
            span_groups = group_spans_by_parent(self._current_spans)
            all_errors = []
            
            for parent_id, spans in span_groups.items():
                if not should_process_group(parent_id):
                    continue
                    
                # Validate pattern within this parent's children
                validator = FlowValidator.from_spans(spans)
                result = validator.validate_pattern(pattern)
                
                if result['valid']:
                    return self  # Success - return immediately
                
                # Store error details for later reporting
                parent_info = get_parent_info(parent_id, self._current_spans)
                error_info = {
                    'parent_info': parent_info,
                    'violations': result['violations'],
                    'suggestions': result.get('suggestions', [])
                }
                all_errors.append(error_info)
            
            # If we reach here, no groups were valid - raise appropriate error
            if not all_errors:
                if parent_filter:
                    raise AssertionError(f"No parent spans found matching filter '{parent_filter}'")
                else:
                    raise AssertionError(f"No parent groups found to validate pattern '{pattern}'")
            
            # Choose error format based on number of errors and parent filter
            single_group = len(all_errors) == 1 or parent_filter is not None
            raise AssertionError(format_validation_error(all_errors, single_group))
                
        except ImportError as e:
            raise AssertionError(f"Flow validation requires visualization module: {e}")
        except AssertionError:
            raise  # Re-raise our own assertion errors
        except Exception as e:
            raise AssertionError(f"Flow pattern validation failed: {pattern}\nError: {str(e)}")
            
        return self
