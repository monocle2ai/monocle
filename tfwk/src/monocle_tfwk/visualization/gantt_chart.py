"""
Gantt chart visualization for Monocle traces.

This module provides functionality to create hierarchical Gantt chart
representations of trace execution flows, showing timing, dependencies,
and hierarchical relationships between spans.
"""
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from monocle_tfwk.assertions.flow_validator import TimelineEvent


class VisualizationMode(Enum):
    """Enumeration of available visualization modes for trace displays."""
    FULL = "full"
    COMPACT = "compact"
    DETAILED = "detailed"


class TraceGanttChart:
    """Creates Gantt chart representations from Monocle traces."""
    
    def __init__(self, spans: List[Any]):
        """
        Initialize with a list of span objects.
        
        Args:
            spans: List of span objects (from Monocle trace data)
        """
        self.spans = spans
        self.timeline_events: List[TimelineEvent] = []
        self.root_events: List[TimelineEvent] = []
        self._event_map: Dict[str, TimelineEvent] = {}
        
    def parse_spans(self) -> List[TimelineEvent]:
        """Parse spans into timeline events and build hierarchical structure."""
        # First pass: create timeline events
        for span in self.spans:
            event = self._create_timeline_event(span)
            self.timeline_events.append(event)
            self._event_map[event.span_id] = event
        
        # Second pass: build parent-child relationships
        for event in self.timeline_events:
            if event.parent_id and event.parent_id in self._event_map:
                parent = self._event_map[event.parent_id]
                parent.children.append(event)
                event.level = parent.level + 1
            else:
                # Root event (no parent)
                self.root_events.append(event)
                
        # Sort events by start time within each level
        self._sort_events()
        
        return self.timeline_events
    
    def _create_timeline_event(self, span: Any) -> TimelineEvent:
        """Create a TimelineEvent from a span object."""
        # Check if this is a ReadableSpan from OpenTelemetry
        if hasattr(span, 'context') and hasattr(span.context, 'span_id') and hasattr(span.context, 'trace_id'):
            # This is a ReadableSpan object
            span_id = span.context.span_id.to_bytes(8, 'big').hex() if span.context.span_id else "unknown"
            trace_id = span.context.trace_id.to_bytes(16, 'big').hex() if span.context.trace_id else "unknown"
            parent_id = span.parent.span_id.to_bytes(8, 'big').hex() if (span.parent and hasattr(span.parent, 'span_id') and span.parent.span_id) else None
            
            name = getattr(span, 'name', '')
            
            # Handle ReadableSpan timestamps (nanoseconds since epoch)
            start_time = datetime.fromtimestamp(span.start_time / 1e9) if span.start_time else None
            end_time = datetime.fromtimestamp(span.end_time / 1e9) if span.end_time else None
            
            # Calculate duration in milliseconds
            duration_ms = (span.end_time - span.start_time) / 1e6 if (span.end_time and span.start_time) else 0.0
            
            # Get attributes safely
            attributes = dict(span.attributes) if span.attributes else {}
            
            # Set span type in attributes if not already present
            if 'span.type' not in attributes:
                attributes['span.type'] = self._classify_span_type(name, attributes)
            
        # Handle dict-based spans
        elif isinstance(span, dict):
            context = span.get('context', {})
            span_id = str(context.get('span_id', ''))
            trace_id = str(context.get('trace_id', ''))
            parent_id = span.get('parent_id')
            
            # Convert string 'null' to None
            if parent_id == 'null' or parent_id == 'None':
                parent_id = None
            elif parent_id:
                parent_id = str(parent_id)
                
            name = span.get('name', '')
            
            # Parse timestamps (assume ISO format for dict spans)
            start_time_str = span.get('start_time', '')
            end_time_str = span.get('end_time', '')
            
            start_time = self._parse_timestamp(start_time_str)
            end_time = self._parse_timestamp(end_time_str)
            
            # Calculate duration
            duration_ms = (end_time - start_time).total_seconds() * 1000 if start_time and end_time else 0
            
            attributes = span.get('attributes', {})
            if attributes is None:
                attributes = {}
                
            # Set span type in attributes if not already present
            if 'span.type' not in attributes:
                attributes['span.type'] = self._classify_span_type(name, attributes)
            
        # Handle other object types
        else:
            span_id = str(getattr(span, 'span_id', ''))
            trace_id = str(getattr(span, 'trace_id', ''))
            parent_id = getattr(span, 'parent_id', None)
            name = getattr(span, 'name', '')
            
            # Try to get timestamps as numeric values first, then as strings
            start_time_raw = getattr(span, 'start_time', '')
            end_time_raw = getattr(span, 'end_time', '')
            
            if isinstance(start_time_raw, (int, float)) and start_time_raw > 1e12:
                # Assume nanoseconds
                start_time = datetime.fromtimestamp(start_time_raw / 1e9)
                end_time = datetime.fromtimestamp(end_time_raw / 1e9) if end_time_raw else None
                duration_ms = (end_time_raw - start_time_raw) / 1e6 if (end_time_raw and start_time_raw) else 0.0
            else:
                # Try parsing as string
                start_time = self._parse_timestamp(str(start_time_raw))
                end_time = self._parse_timestamp(str(end_time_raw))
                duration_ms = (end_time - start_time).total_seconds() * 1000 if start_time and end_time else 0
            
            attributes = getattr(span, 'attributes', {})
            if attributes is None:
                attributes = {}
                
            # Set span type in attributes if not already present
            if 'span.type' not in attributes:
                attributes['span.type'] = self._classify_span_type(name, attributes)
            
        return TimelineEvent(
            span_id=span_id,
            parent_id=parent_id,
            trace_id=trace_id,
            name=name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            attributes=attributes
        )
    
    def _classify_span_type(self, name: str, attributes: Dict[str, Any]) -> str:
        """Classify span type based on name and attributes."""
        name_lower = name.lower()
        
        # HTTP-related spans
        if 'fastapi.request' in name_lower:
            return 'http.request'
        elif 'fastapi.response' in name_lower:
            return 'http.response' 
        elif 'http' in name_lower or 'fastapi' in name_lower:
            return 'http.process'
            
        # Workflow orchestration
        elif 'workflow' in name_lower:
            return 'workflow'
            
        # Business logic operations - be more specific based on actual span names
        elif 'validate_user_data' in name_lower:
            return 'validation'
        elif 'calculate_user_profile_score' in name_lower:
            return 'computation'
        elif 'send_notification' in name_lower:
            return 'notification'
        elif 'audit_log_operation' in name_lower:
            return 'audit'
            
        # General patterns for other business logic
        elif 'validate' in name_lower:
            return 'validation'
        elif 'calculate' in name_lower or 'compute' in name_lower:
            return 'computation'
        elif 'send' in name_lower or 'notification' in name_lower:
            return 'notification'
        elif 'audit' in name_lower or 'log' in name_lower:
            return 'audit'
            
        # Database operations
        elif any(db in name_lower for db in ['database', 'db', 'query', 'select', 'insert', 'update']):
            return 'database'
            
        # External service calls  
        elif any(svc in name_lower for svc in ['api', 'service', 'client', 'call']):
            return 'external'
            
        # Default fallback
        else:
            return 'business_logic'
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime object."""
        if not timestamp_str:
            return None
        try:
            # Handle different timestamp formats
            if timestamp_str.endswith('Z'):
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(timestamp_str)
        except (ValueError, AttributeError):
            return None
    
    def _sort_events(self):
        """Sort events by start time at each level."""
        def sort_children(event: TimelineEvent):
            # Sort with None-safe key
            event.children.sort(key=lambda e: e.start_time if e.start_time else datetime.min)
            for child in event.children:
                sort_children(child)
        
        # Sort with None-safe key
        self.root_events.sort(key=lambda e: e.start_time if e.start_time else datetime.min)
        for root in self.root_events:
            sort_children(root)
    
    def generate_gantt_text(self, show_attributes: bool = True) -> str:
        """Generate a text-based Gantt chart representation."""
        if not self.timeline_events:
            self.parse_spans()
            
        lines = []
        lines.append("=== Trace Gantt Chart ===")
        lines.append("")
        
        # Find time bounds
        all_events = self.timeline_events
        if not all_events:
            return "No timeline events found"
            
        # Get events with valid timestamps
        valid_start_times = [e.start_time for e in all_events if e.start_time]
        valid_end_times = [e.end_time for e in all_events if e.end_time]
        
        if not valid_start_times or not valid_end_times:
            return "No valid timestamps found in events"
            
        min_time = min(valid_start_times)
        max_time = max(valid_end_times)
        total_duration = (max_time - min_time).total_seconds() * 1000
        
        lines.append(f"Total trace duration: {total_duration:.2f}ms")
        lines.append(f"Start time: {min_time.isoformat()}")
        lines.append(f"End time: {max_time.isoformat()}")
        lines.append("")
        
        # Generate chart
        for root in self.root_events:
            lines.extend(self._render_event_tree_with_attributes(root, show_attributes=show_attributes))
            lines.append("")
            
        return "\n".join(lines)
    
    def generate_attributes_text(self, show_attributes: bool = True) -> str:
        """Generate a text-based tree representation with span attributes."""
        if not self.timeline_events:
            self.parse_spans()
            
        lines = []
        lines.append("=== Trace Attributes Tree ===")
        lines.append("")
        
        all_events = self.timeline_events
        if not all_events:
            return "No timeline events found"
        
        lines.append(f"Total spans: {len(all_events)}")
        lines.append(f"Root spans: {len(self.root_events)}")
        lines.append("")
        
        # Generate attribute tree
        for root in self.root_events:
            lines.extend(self._render_event_tree_with_attributes(root, show_attributes=show_attributes))
            lines.append("")
            
        return "\n".join(lines)
    
    def _render_event_tree_with_attributes(self, event: TimelineEvent, prefix: str = "", show_attributes: bool = True) -> List[str]:
        """Render an event and its children as text tree with attributes instead of timing."""
        lines = []
        
        # Format event info with span name, type and id
        event_header = f"{prefix}├─ ({event.span_type} | {event.span_id}) {event.name}"
        lines.append(event_header)
        
        # Only show attributes if requested
        if show_attributes:
            # Build attributes dictionary string
            attr_dict = {}
            if event.attributes:
                for key, value in event.attributes.items():
                    # Skip internal or verbose attributes for cleaner display, including span.type since it's shown above
                    if key.startswith('_') or key in ['span_source', 'monocle_apptrace.version', 'monocle_apptrace.language', 'span.type']:
                        continue
                        
                    # Don't truncate values - show full content
                    value_str = str(value)
                    attr_dict[key] = value_str
            
            # Format attributes as individual lines
            if attr_dict:
                for key, value in attr_dict.items():
                    attr_line = f"{prefix}│    {key}: {value}"
                    lines.append(attr_line)
                # Add empty line after attributes for readability
                lines.append(f"{prefix}│")
        
        # Render children
        for i, child in enumerate(event.children):
            child_prefix = prefix + ("│  " if i < len(event.children) - 1 else "   ")
            lines.extend(self._render_event_tree_with_attributes(child, child_prefix, show_attributes))
            
        return lines
    
    def generate_mermaid_gantt(self) -> str:
        """Generate a Mermaid.js Gantt chart representation."""
        if not self.timeline_events:
            self.parse_spans()
            
        lines = []
        lines.append("gantt")
        lines.append("    title Trace Execution Flow")
        lines.append("    dateFormat X")
        lines.append("    axisFormat %s")
        lines.append("")
        
        # Find time bounds
        all_events = self.timeline_events
        if not all_events:
            return "gantt\n    title Empty Trace"
            
        min_time = min(e.start_time for e in all_events if e.start_time)
        
        # Group by workflow/section
        workflows = {}
        for event in all_events:
            workflow = event.workflow_name or "Unknown"
            if workflow not in workflows:
                workflows[workflow] = []
            workflows[workflow].append(event)
        
        for workflow_name, events in workflows.items():
            lines.append(f"    section {workflow_name}")
            
            for event in sorted(events, key=lambda e: e.start_time):
                if not event.start_time:
                    continue
                    
                start_ms = int((event.start_time - min_time).total_seconds() * 1000)
                end_ms = start_ms + int(event.duration_ms)
                
                # Clean name for Mermaid
                clean_name = event.name.replace(".", "_").replace(" ", "_")[:30]
                
                lines.append(f"    {clean_name}    :{start_ms}, {end_ms}")
        
        return "\n".join(lines)
    
    def get_critical_path(self) -> List[TimelineEvent]:
        """Find the critical path (longest duration chain) through the trace."""
        if not self.timeline_events:
            self.parse_spans()
            
        def find_longest_path(event: TimelineEvent) -> Tuple[float, List[TimelineEvent]]:
            if not event.children:
                return event.duration_ms, [event]
                
            max_duration = 0
            longest_path = [event]
            
            for child in event.children:
                child_duration, child_path = find_longest_path(child)
                total_duration = event.duration_ms + child_duration
                
                if total_duration > max_duration:
                    max_duration = total_duration
                    longest_path = [event] + child_path
                    
            return max_duration, longest_path
        
        # Find longest path among all root events
        max_duration = 0
        critical_path = []
        
        for root in self.root_events:
            duration, path = find_longest_path(root)
            if duration > max_duration:
                max_duration = duration
                critical_path = path
                
        return critical_path
    
    def to_json(self) -> str:
        """Export timeline data as JSON."""
        if not self.timeline_events:
            self.parse_spans()
            
        data = {
            "timeline_events": [event.to_dict() for event in self.timeline_events],
            "root_events": [event.to_dict() for event in self.root_events],
            "critical_path": [event.to_dict() for event in self.get_critical_path()]
        }
        
        return json.dumps(data, indent=2, default=str)
    
    def generate_visualization_report(self, mode: VisualizationMode = VisualizationMode.FULL) -> str:
        """
        Generate a comprehensive visualization and validation report.
        
        Args:
            mode: Visualization mode that determines what to include in the report
            
        Returns:
            Formatted report string
            
        Example:
            ```python
            gantt = TraceGanttChart(spans)
            
            # Full report with all features
            full_report = gantt.generate_visualization_report(VisualizationMode.FULL)
            
            # Compact report - just timeline without attributes
            compact_report = gantt.generate_visualization_report(VisualizationMode.COMPACT)
            
            # Detailed report - attributes-focused
            detailed_report = gantt.generate_visualization_report(VisualizationMode.DETAILED)
            ```
        """
        # Set parameters based on mode
        if mode == VisualizationMode.COMPACT:
            show_attributes = False
            include_critical_path = False
            include_mermaid = False
        elif mode == VisualizationMode.DETAILED:
            show_attributes = True
            include_critical_path = False
            include_mermaid = False
        else:  # VisualizationMode.FULL
            show_attributes = True
            include_critical_path = True
            include_mermaid = True
        
        lines = []
        lines.append("=" * 60)
        lines.append("MONOCLE TRACE VISUALIZATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Timeline visualization
        lines.append("TIMELINE VISUALIZATION")
        lines.append("-" * 30)
        lines.append(self.generate_gantt_text(show_attributes=show_attributes))
        lines.append("")
        
        # Critical path analysis
        if include_critical_path:
            critical_path = self.get_critical_path()
            if critical_path:
                lines.append("CRITICAL PATH ANALYSIS")
                lines.append("-" * 30)
                total_critical_duration = sum(event.duration_ms for event in critical_path)
                lines.append(f"Critical path duration: {total_critical_duration:.2f}ms")
                lines.append("Critical path events:")
                for event in critical_path:
                    lines.append(f"  → {event.name} ({event.duration_ms:.1f}ms)")
                lines.append("")
        
        # Mermaid diagram
        if include_mermaid:
            lines.append("MERMAID GANTT DIAGRAM")
            lines.append("-" * 30)
            lines.append("```mermaid")
            lines.append(self.generate_mermaid_gantt())
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    


# Utility functions for backward compatibility and convenience

def create_gantt_from_json_traces(trace_json: str) -> TraceGanttChart:
    traces = json.loads(trace_json)
    return TraceGanttChart(traces)
