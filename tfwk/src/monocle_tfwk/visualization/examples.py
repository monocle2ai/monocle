"""
Example usage and integration guide for Monocle trace visualization.

This module demonstrates how to use the Gantt chart visualization and flow
validation tools with real trace data.
"""
import json
from typing import Any, Dict, List

from monocle_tfwk.assertions.flow_validator import FlowPattern, FlowValidator
from monocle_tfwk.visualization.gantt_chart import TraceGanttChart


def create_gantt_from_json_traces(trace_json: str) -> TraceGanttChart:
    """
    Create a Gantt chart from JSON trace data.
    
    Args:
        trace_json: JSON string containing array of trace spans
        
    Returns:
        TraceGanttChart instance with parsed timeline events
    
    Example:
        ```python
        trace_data = '''[
            {
                "name": "workflow",
                "context": {"trace_id": "0x123", "span_id": "0x456"},
                "parent_id": null,
                "start_time": "2024-01-01T10:00:00Z",
                "end_time": "2024-01-01T10:00:05Z",
                "attributes": {"span.type": "workflow"}
            },
            {
                "name": "openai.chat.completion",
                "context": {"trace_id": "0x123", "span_id": "0x789"},
                "parent_id": "0x456",
                "start_time": "2024-01-01T10:00:01Z",
                "end_time": "2024-01-01T10:00:04Z",
                "attributes": {"span.type": "inference"}
            }
        ]'''
        
        gantt = create_gantt_from_json_traces(trace_data)
        print(gantt.generate_gantt_text())
        ```
    """
    traces = json.loads(trace_json)
    return TraceGanttChart(traces)


def validate_common_patterns(gantt_chart: TraceGanttChart) -> Dict[str, Any]:
    """
    Validate common execution patterns against trace data.
    
    Args:
        gantt_chart: Parsed trace Gantt chart
        
    Returns:
        Dictionary with validation results for common patterns
        
    Example:
        ```python
        gantt = create_gantt_from_json_traces(trace_data)
        results = validate_common_patterns(gantt)
        
        for pattern, result in results.items():
            print(f"{pattern}: {'✓' if result['valid'] else '✗'}")
        ```
    """
    validator = FlowValidator(gantt_chart)
    
    common_patterns = [
        FlowPattern("workflow_to_inference", "workflow -> inference", 
                   "Workflow should precede inference operations"),
        FlowPattern("retrieval_before_generation", "retrieval -> inference",
                   "Information retrieval should happen before generation"),
        FlowPattern("parallel_processing", "inference || inference",
                   "Multiple inference operations can run in parallel"),
        FlowPattern("agent_workflow", "agent.* -> tool.* -> agent.*",
                   "Agent should use tools and then continue processing")
    ]
    
    results = {}
    for pattern in common_patterns:
        results[pattern.name] = validator.validate_pattern(pattern)
        
    return results


def generate_visualization_report(gantt_chart: TraceGanttChart, 
                                patterns: List[str] = None) -> str:
    """
    Generate a comprehensive visualization and validation report.
    
    Args:
        gantt_chart: Parsed trace Gantt chart
        patterns: Optional list of pattern strings to validate
        
    Returns:
        Formatted report string
        
    Example:
        ```python
        gantt = create_gantt_from_json_traces(trace_data)
        report = generate_visualization_report(
            gantt, 
            ["workflow -> inference", "retrieval -> generation"]
        )
        print(report)
        ```
    """
    lines = []
    lines.append("=" * 60)
    lines.append("MONOCLE TRACE VISUALIZATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Timeline visualization
    lines.append("TIMELINE VISUALIZATION")
    lines.append("-" * 30)
    lines.append(gantt_chart.generate_gantt_text())
    lines.append("")
    
    # Critical path analysis
    critical_path = gantt_chart.get_critical_path()
    if critical_path:
        lines.append("CRITICAL PATH ANALYSIS")
        lines.append("-" * 30)
        total_critical_duration = sum(event.duration_ms for event in critical_path)
        lines.append(f"Critical path duration: {total_critical_duration:.2f}ms")
        lines.append("Critical path events:")
        for event in critical_path:
            lines.append(f"  → {event.name} ({event.duration_ms:.1f}ms)")
        lines.append("")
    
    # Flow validation
    if patterns:
        validator = FlowValidator(gantt_chart)
        flow_patterns = [FlowPattern(f"pattern_{i}", p) for i, p in enumerate(patterns)]
        
        lines.append("FLOW VALIDATION")
        lines.append("-" * 30)
        lines.append(validator.generate_flow_report(flow_patterns))
        lines.append("")
    
    # Suggested patterns
    validator = FlowValidator(gantt_chart)
    suggested = validator.suggest_patterns()
    if suggested:
        lines.append("SUGGESTED PATTERNS")
        lines.append("-" * 30)
        for pattern in suggested:
            lines.append(f"Pattern: {pattern.name}")
            lines.append(f"  Expression: {pattern.pattern}")
            lines.append(f"  Description: {pattern.description}")
        lines.append("")
    
    # Mermaid diagram
    lines.append("MERMAID GANTT DIAGRAM")
    lines.append("-" * 30)
    lines.append("```mermaid")
    lines.append(gantt_chart.generate_mermaid_gantt())
    lines.append("```")
    lines.append("")
    
    return "\n".join(lines)


# Example usage patterns
EXAMPLE_PATTERNS = {
    "simple_sequence": "workflow -> inference",
    "rag_pattern": "retrieval -> inference -> response",
    "agent_tool_usage": "agent -> tool -> agent",
    "parallel_inference": "inference || inference",
    "retry_pattern": "inference -> inference?",
    "multi_agent": "agent.* -> agent.*",
    "database_query": "workflow -> database -> inference",
    "streaming_response": "inference -> stream.*"
}


def get_example_patterns() -> Dict[str, FlowPattern]:
    """Get pre-defined example patterns for common use cases."""
    patterns = {}
    
    descriptions = {
        "simple_sequence": "Basic workflow followed by inference",
        "rag_pattern": "Retrieval-Augmented Generation pattern",
        "agent_tool_usage": "Agent using tools and continuing",
        "parallel_inference": "Multiple parallel inference operations",
        "retry_pattern": "Inference with optional retry",
        "multi_agent": "Multi-agent collaboration",
        "database_query": "Database query before inference",
        "streaming_response": "Inference with streaming output"
    }
    
    for name, pattern_str in EXAMPLE_PATTERNS.items():
        patterns[name] = FlowPattern(
            name=name,
            pattern=pattern_str,
            description=descriptions.get(name, "")
        )
    
    return patterns


def demo_visualization():
    """Demonstrate the visualization capabilities with sample data."""
    
    # Sample trace data
    sample_traces = [
        {
            "name": "workflow",
            "context": {
                "trace_id": "0x1234567890abcdef",
                "span_id": "0x1111111111111111"
            },
            "parent_id": None,
            "start_time": "2024-01-01T10:00:00.000Z",
            "end_time": "2024-01-01T10:00:05.000Z",
            "attributes": {
                "span.type": "workflow",
                "workflow.name": "rag_demo"
            }
        },
        {
            "name": "retrieval.query",
            "context": {
                "trace_id": "0x1234567890abcdef",
                "span_id": "0x2222222222222222"
            },
            "parent_id": "0x1111111111111111",
            "start_time": "2024-01-01T10:00:00.500Z",
            "end_time": "2024-01-01T10:00:02.000Z",
            "attributes": {
                "span.type": "retrieval",
                "workflow.name": "rag_demo"
            }
        },
        {
            "name": "openai.chat.completion",
            "context": {
                "trace_id": "0x1234567890abcdef",
                "span_id": "0x3333333333333333"
            },
            "parent_id": "0x1111111111111111",
            "start_time": "2024-01-01T10:00:02.100Z",
            "end_time": "2024-01-01T10:00:04.500Z",
            "attributes": {
                "span.type": "inference",
                "workflow.name": "rag_demo"
            }
        }
    ]
    
    # Create Gantt chart
    gantt = TraceGanttChart(sample_traces)
    
    # Generate report with validation
    patterns = ["workflow -> retrieval", "retrieval -> inference"]
    report = generate_visualization_report(gantt, patterns)
    
    print(report)
    return gantt


if __name__ == "__main__":
    demo_visualization()