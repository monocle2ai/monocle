import datetime
import os
import json
from typing import Any, Dict
from opentelemetry.sdk.trace import Span, ReadableSpan, Status, StatusCode, Event, Resource
from opentelemetry import trace as trace_api

class JSONSpanLoader:
    """Utility class to load spans from JSON files and convert them to ReadableSpan instances."""

    @staticmethod
    def load_spans(trace_file_path: str) -> list["ReadableSpan"]:
        current_script_path = os.path.abspath(__file__)
        spans = JSONSpanLoader.from_json(os.path.join(os.path.dirname(current_script_path), trace_file_path))
        return spans

    @staticmethod
    def from_json(json_file_path: str) -> list["ReadableSpan"]:
        """Create a ReadableSpan instance from a JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing span data.
            
        Returns:
            A ReadableSpan instance.
        """
        span_list = []
        with open(json_file_path, 'r') as f:
            span_data = json.load(f)

            for item in span_data:
                span = JSONSpanLoader._from_dict(span_data=item)
                span_list.append(span)

        return span_list

    @staticmethod
    def _from_dict(span_data: Dict[str, Any]) -> "ReadableSpan":
        """Create a ReadableSpan instance from a dictionary.
        
        Args:
            span_data: Dictionary containing span data.
            
        Returns:
            A ReadableSpan instance.
        """
        # Parse context
        context = None
        if span_data.get("context"):
            context = JSONSpanLoader._parse_context(span_data["context"])

        # Parse parent context
        parent = None
        if span_data.get("parent_id"):
            # Extract span_id from formatted parent_id (e.g., "0x1234567890abcdef")
            parent_span_id = int(span_data["parent_id"], 16)
            # We need the trace_id from context to create parent context
            if context:
                parent = trace_api.SpanContext(
                    trace_id=context.trace_id,
                    span_id=parent_span_id,
                    is_remote=True,  # Assume remote for reconstructed parent
                    trace_flags=trace_api.TraceFlags.DEFAULT,
                    trace_state=None
                )
        
        # Parse timestamps
        start_time = None
        if span_data.get("start_time"):
            start_time = JSONSpanLoader._iso_str_to_ns(span_data["start_time"])

        end_time = None
        if span_data.get("end_time"):
            end_time = JSONSpanLoader._iso_str_to_ns(span_data["end_time"])

        # Parse status
        status = Status(StatusCode.UNSET)
        if span_data.get("status"):
            status_data = span_data["status"]
            status_code = getattr(StatusCode, status_data["status_code"], StatusCode.UNSET)
            description = status_data.get("description")
            status = Status(status_code, description)
        
        # Parse kind
        kind = trace_api.SpanKind.INTERNAL
        if span_data.get("kind"):
            kind_str = span_data["kind"].replace("SpanKind.", "")
            kind = getattr(trace_api.SpanKind, kind_str, trace_api.SpanKind.INTERNAL)
        
        # Parse events
        events = []
        if span_data.get("events"):
            for event_data in span_data["events"]:
                timestamp = JSONSpanLoader._iso_str_to_ns(event_data["timestamp"])
                event = Event(
                    name=event_data["name"],
                    attributes=event_data.get("attributes"),
                    timestamp=timestamp
                )
                events.append(event)
        
        # Parse links
        links = []
        if span_data.get("links"):
            for link_data in span_data["links"]:
                link_context = JSONSpanLoader._parse_context(link_data["context"])
                link = trace_api.Link(
                    context=link_context,
                    attributes=link_data.get("attributes")
                )
                links.append(link)
        
        # Parse resource
        resource = None
        if span_data.get("resource"):
            resource = Resource.create(span_data["resource"].get("attributes", {}))
        
        return ReadableSpan(
            name=span_data["name"],
            context=context,
            parent=parent,
            resource=resource,
            attributes=span_data.get("attributes"),
            events=events,
            links=links,
            kind=kind,
            status=status,
            start_time=start_time,
            end_time=end_time
        )
    
    @staticmethod
    def _parse_context(context_data: Dict[str, str]) -> trace_api.SpanContext:
        """Parse context from dictionary format back to SpanContext."""
        # Parse trace_id and span_id from hex format (e.g., "0x1234567890abcdef")
        trace_id = int(context_data["trace_id"], 16)
        span_id = int(context_data["span_id"], 16)
        
        # Parse trace_state (it was stored as repr(), so we need to handle it carefully)
        trace_state = None
        trace_state_str = context_data.get("trace_state", "None")
        if trace_state_str != "None" and trace_state_str:
            # The trace_state was stored as repr(), so it might be a string representation
            # For now, we'll leave it as None since parsing repr() output is complex
            pass
        
        return trace_api.SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=trace_api.TraceFlags.DEFAULT,
            trace_state=trace_state
        )
    
    @staticmethod
    def _iso_str_to_ns(iso_str: str) -> int:
        """Convert ISO 8601 string back to nanoseconds since epoch."""
        # Parse the ISO format: "2023-01-01T12:00:00.123456Z"
        dt = datetime.datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return int(dt.timestamp() * 1e9)