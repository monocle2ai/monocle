import datetime
import json
import os
import glob
from typing import Any, Dict, List, Optional
from opentelemetry.sdk.trace import ReadableSpan, Status, StatusCode, Event, Resource
from opentelemetry import trace as trace_api
from monocle_apptrace.exporters.file_exporter import DEFAULT_TRACE_FOLDER


class JSONSpanLoader:
    """Utility class to load spans from JSON trace files."""

    @staticmethod
    def from_json(json_file_path: str) -> List[ReadableSpan]:
        """Load spans from a JSON file.

        Args:
            json_file_path: Absolute path to the JSON file containing span data.

        Returns:
            A list of ReadableSpan instances.
        """
        span_list = []
        with open(json_file_path, 'r') as f:
            span_data = json.load(f)
            for item in span_data:
                span = JSONSpanLoader._from_dict(span_data=item)
                span_list.append(span)
        return span_list

    @staticmethod
    def find_trace_file(trace_id: str, trace_dir: Optional[str] = None) -> Optional[str]:
        """Find a trace file matching the given trace_id.

        The file naming convention is:
            monocle_trace_{service_name}_{trace_id}_{timestamp}.json

        Args:
            trace_id: The trace ID to search for (hex string without 0x prefix).
            trace_dir: Directory to search in. Defaults to .monocle/test_traces.

        Returns:
            The file path if found, None otherwise.
        """
        if trace_dir is None:
            trace_dir = os.path.join(".", DEFAULT_TRACE_FOLDER, "test_traces")

        # Strip 0x prefix if present
        trace_id = trace_id.replace("0x", "")

        pattern = os.path.join(trace_dir, f"*_{trace_id}_*.json")
        matches = glob.glob(pattern)
        if matches:
            # Return the most recent file if multiple matches
            return max(matches, key=os.path.getmtime)
        return None

    @staticmethod
    def _from_dict(span_data: Dict[str, Any]) -> ReadableSpan:
        """Create a ReadableSpan instance from a dictionary."""
        # Parse context
        context = None
        if span_data.get("context"):
            context = JSONSpanLoader._parse_context(span_data["context"])

        # Parse parent context
        parent = None
        if span_data.get("parent_id"):
            parent_span_id = int(span_data["parent_id"], 16)
            if context:
                parent = trace_api.SpanContext(
                    trace_id=context.trace_id,
                    span_id=parent_span_id,
                    is_remote=True,
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
        """Parse span context from dictionary."""
        trace_id = int(context_data["trace_id"], 16)
        span_id = int(context_data["span_id"], 16)
        return trace_api.SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=trace_api.TraceFlags.DEFAULT,
            trace_state=None
        )

    @staticmethod
    def _iso_str_to_ns(iso_str: str) -> int:
        """Convert ISO format timestamp string to nanoseconds."""
        dt = datetime.datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return int(dt.timestamp() * 1e9)
