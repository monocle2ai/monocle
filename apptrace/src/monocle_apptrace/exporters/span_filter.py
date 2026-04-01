"""
Span filtering module for Monocle.

Provides flexible filtering and projection of span data based on:
- Span types (e.g., inference, retrieval, agentic.*)
- Attributes (e.g., entity.1.name, scope.*)
- Events and their attributes (e.g., metadata.completion_tokens)
"""

import logging
from typing import Dict, List, Optional, Any, Sequence
from opentelemetry.sdk.trace import ReadableSpan
import json

logger = logging.getLogger(__name__)


class SpanFilter:
    """
    Filters and projects span data based on configuration.
    
    Allows you to:
    1. Include only specific span types
    2. Project only specific attributes
    3. Project only specific events and their attributes
    
    Example:
        config = {
            "span_types_to_include": ["inference", "inference.framework"],
            "fields_to_include": {
                "attributes": ["entity.1.name", "entity.2.name", "scope.customer_id"],
                "events": [
                    {"name": "metadata", "attributes": ["completion_tokens", "prompt_tokens"]},
                    {"name": "data.output", "attributes": ["response"]}
                ]
            }
        }
        
        span_filter = SpanFilter(config)
        filtered_data = span_filter.filter(span)
    """
    
    def __init__(self, filter_config: Dict[str, Any]):
        """
        Initialize the span filter with configuration.
        
        Args:
            filter_config: Dictionary containing filter configuration:
                - span_types_to_include (List[str]): Span types to include (e.g., ["inference", "retrieval"])
                                                      Supports wildcards (e.g., "inference.*" matches "inference.framework")
                - fields_to_include (Dict): Fields to project from matching spans:
                    - attributes (List[str]): Attribute keys to include (e.g., ["entity.1.name", "scope.*"])
                    - events (List[Dict]): Event configurations, each with:
                        - name (str): Event name (e.g., "metadata")
                        - attributes (List[str]): Event attribute keys to include
                - mode (str, optional): "include" (default) or "exclude" - whether to include or exclude specified types
        """
        self.config = filter_config
        self.span_types_to_include = filter_config.get("span_types_to_include", [])
        self.fields_to_include = filter_config.get("fields_to_include", {})
        self.mode = filter_config.get("mode", "include")  # "include" or "exclude"
        
        # Parse attribute patterns (support wildcards like "scope.*")
        self.attribute_patterns = self.fields_to_include.get("attributes", []) if isinstance(self.fields_to_include, dict) else []
        self.event_configs = self.fields_to_include.get("events", []) if isinstance(self.fields_to_include, dict) else []
        
        # Validate configuration after all attributes are set
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the filter configuration."""
        if not isinstance(self.span_types_to_include, list):
            raise ValueError("span_types_to_include must be a list")
        
        if not isinstance(self.fields_to_include, dict):
            raise ValueError("fields_to_include must be a dictionary")
        
        if self.mode not in ["include", "exclude"]:
            raise ValueError("mode must be 'include' or 'exclude'")
        
        # Validate event configs
        for event_config in self.event_configs:
            if not isinstance(event_config, dict):
                raise ValueError("Each event config must be a dictionary")
            if "name" not in event_config:
                raise ValueError("Each event config must have a 'name' field")
    
    def should_include_span(self, span: ReadableSpan) -> bool:
        """
        Determine if a span should be included based on its type.
        
        Args:
            span: The span to check
        
        Returns:
            True if the span should be included, False otherwise
        """
        if not self.span_types_to_include:
            # If no types specified, include all
            return True
        
        span_type = span.attributes.get("span.type", "")
        
        for pattern in self.span_types_to_include:
            if self._matches_pattern(span_type, pattern):
                return self.mode == "include"
        
        # If no match found
        return self.mode == "exclude"
    
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """
        Check if a value matches a pattern (supports wildcards).
        
        Args:
            value: The value to check
            pattern: The pattern (supports * wildcard)
        
        Returns:
            True if the value matches the pattern
        """
        if pattern == "*":
            return True
        
        if "*" in pattern:
            # Simple wildcard matching
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                return value.startswith(prefix)
            elif pattern.startswith("*."):
                suffix = pattern[2:]
                return value.endswith(suffix)
            else:
                # More complex patterns - convert to regex-like
                import re
                regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
                return re.match(f"^{regex_pattern}$", value) is not None
        
        return value == pattern
    
    def filter(self, span: ReadableSpan) -> Optional[Dict[str, Any]]:
        """
        Filter and project span data according to configuration.
        
        Args:
            span: The span to filter
        
        Returns:
            Filtered span as a dictionary, or None if span should be excluded
        """
        # Check if span type should be included
        if not self.should_include_span(span):
            return None
        
        # Convert span to dict for manipulation
        try:
            span_dict = json.loads(span.to_json())
        except Exception as e:
            logger.warning(f"Failed to convert span to JSON: {e}")
            return None
        
        # If no field filtering specified, return the full span
        if not self.fields_to_include:
            return span_dict
        
        # Create filtered span dict
        filtered_span = {
            "name": span_dict.get("name"),
            "context": span_dict.get("context"),
            "kind": span_dict.get("kind"),
            "parent_id": span_dict.get("parent_id"),
            "start_time": span_dict.get("start_time"),
            "end_time": span_dict.get("end_time"),
            "status": span_dict.get("status"),
        }
        
        # Filter attributes
        if self.attribute_patterns:
            filtered_span["attributes"] = self._filter_attributes(span_dict.get("attributes", {}))
        else:
            # If no attribute filtering, include all attributes
            filtered_span["attributes"] = span_dict.get("attributes", {})
        
        # Filter events
        if self.event_configs:
            filtered_span["events"] = self._filter_events(span_dict.get("events", []))
        else:
            # If no event filtering, include all events
            filtered_span["events"] = span_dict.get("events", [])
        
        # Include resource info (useful for context)
        filtered_span["resource"] = span_dict.get("resource", {})
        
        return filtered_span
    
    def _filter_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter attributes based on configured patterns.
        
        Args:
            attributes: Original attributes dictionary
        
        Returns:
            Filtered attributes dictionary
        """
        if not self.attribute_patterns:
            return attributes
        
        filtered = {}
        
        for pattern in self.attribute_patterns:
            for key, value in attributes.items():
                if self._matches_pattern(key, pattern) and key not in filtered:
                    filtered[key] = value
        
        return filtered
    
    def _filter_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter events based on configured event specs.
        
        Args:
            events: Original events list
        
        Returns:
            Filtered events list
        """
        if not self.event_configs:
            return events
        
        filtered_events = []
        
        for event in events:
            event_name = event.get("name", "")
            
            # Find matching event config
            matching_config = None
            for config in self.event_configs:
                if self._matches_pattern(event_name, config["name"]):
                    matching_config = config
                    break
            
            if matching_config:
                # Filter event attributes
                event_attributes = event.get("attributes", {})
                
                if "attributes" in matching_config:
                    filtered_attributes = {}
                    for attr_pattern in matching_config["attributes"]:
                        for key, value in event_attributes.items():
                            if self._matches_pattern(key, attr_pattern) and key not in filtered_attributes:
                                filtered_attributes[key] = value
                    
                    filtered_events.append({
                        "name": event_name,
                        "timestamp": event.get("timestamp"),
                        "attributes": filtered_attributes
                    })
                else:
                    # Include entire event if no attribute filtering
                    filtered_events.append(event)
        
        return filtered_events
    
    def filter_multiple(self, spans: Sequence[ReadableSpan]) -> List[Dict[str, Any]]:
        """
        Filter multiple spans.
        
        Args:
            spans: Sequence of spans to filter
        
        Returns:
            List of filtered span dictionaries (excludes spans that don't match filters)
        """
        filtered_spans = []
        
        for span in spans:
            filtered = self.filter(span)
            if filtered is not None:
                filtered_spans.append(filtered)
        
        return filtered_spans


class FilteredReadableSpan:
    """
    Wrapper around ReadableSpan that returns filtered JSON.
    
    This allows the filtered span to be passed through the standard
    SpanExporter interface while controlling what gets serialized.
    """
    
    def __init__(self, original_span: ReadableSpan, filtered_data: Dict[str, Any]):
        """
        Initialize the wrapper.
        
        Args:
            original_span: The original ReadableSpan object
            filtered_data: The filtered/projected span data dictionary
        """
        self._original_span = original_span
        self._filtered_data = filtered_data
    
    def to_json(self, **kwargs) -> str:
        """
        Return filtered JSON instead of full span.
        
        Args:
            **kwargs: Ignored (for compatibility with ReadableSpan.to_json)
        
        Returns:
            JSON string of filtered span data
        """
        # Respect indent if provided
        indent = kwargs.get('indent')
        return json.dumps(self._filtered_data, indent=indent)
    
    def __getattr__(self, name):
        """
        Delegate all other attributes to the original span.
        
        This ensures the wrapper behaves like a ReadableSpan for
        all properties except to_json().
        """
        return getattr(self._original_span, name)


class FilteredSpanExporter:
    """
    Wrapper exporter that filters spans before passing to wrapped exporter.
    
    Example:
        from monocle_apptrace.exporters.file_exporter import FileSpanExporter
        from monocle_apptrace.exporters.span_filter import FilteredSpanExporter, SpanFilter
        
        filter_config = {
            "span_types_to_include": ["inference"],
            "fields_to_include": {
                "attributes": ["entity.1.name", "scope.*"],
                "events": [{"name": "metadata", "attributes": ["completion_tokens"]}]
            }
        }
        
        base_exporter = FileSpanExporter()
        filtered_exporter = FilteredSpanExporter(
            base_exporter=base_exporter,
            span_filter=SpanFilter(filter_config)
        )
    """
    
    def __init__(self, base_exporter, span_filter: SpanFilter):
        """
        Initialize the filtered exporter.
        
        Args:
            base_exporter: The underlying exporter to wrap
            span_filter: SpanFilter instance to use for filtering
        """
        self.base_exporter = base_exporter
        self.span_filter = span_filter
    
    def export(self, spans: Sequence[ReadableSpan]):
        """
        Export filtered and projected spans.
        
        Applies both span type filtering and field projection,
        then wraps the result in FilteredReadableSpan objects.
        """
        filtered_spans = []
        
        for span in spans:
            # Apply full filtering (type + field projection)
            filtered_data = self.span_filter.filter(span)
            
            if filtered_data is not None:
                # Wrap the span so it returns filtered data when serialized
                wrapped_span = FilteredReadableSpan(span, filtered_data)
                filtered_spans.append(wrapped_span)
        
        if filtered_spans:
            return self.base_exporter.export(filtered_spans)
        
        # If no spans passed filter, return success (nothing to export)
        from opentelemetry.sdk.trace.export import SpanExportResult
        return SpanExportResult.SUCCESS
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Delegate flush to base exporter."""
        return self.base_exporter.force_flush(timeout_millis)
    
    def shutdown(self) -> None:
        """Delegate shutdown to base exporter."""
        return self.base_exporter.shutdown()
