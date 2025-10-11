"""
Integration test for trace visualization functionality.

This test demonstrates the Gantt chart and flow validation features
using real trace data from the Monocle test suite.
"""
import json

from monocle_tfwk.assertions.flow_validator import FlowValidator
from monocle_tfwk.visualization.examples import generate_visualization_report
from monocle_tfwk.visualization.gantt_chart import TraceGanttChart


def create_sample_traces():
    """Create sample trace data for testing."""
    return [
        {
            "name": "workflow",
            "context": {
                "trace_id": "0x1234567890abcdef",
                "span_id": "0x1111111111111111",
                "trace_state": "[]"
            },
            "kind": "SpanKind.INTERNAL",
            "parent_id": None,
            "start_time": "2024-01-01T10:00:00.000Z",
            "end_time": "2024-01-01T10:00:05.000Z",
            "status": {"status_code": "OK"},
            "attributes": {
                "span.type": "workflow",
                "workflow.name": "test_workflow",
                "entity.1.name": "test_app",
                "entity.1.type": "workflow.generic"
            },
            "events": [],
            "links": []
        },
        {
            "name": "retrieval.vector_store",
            "context": {
                "trace_id": "0x1234567890abcdef",
                "span_id": "0x2222222222222222",
                "trace_state": "[]"
            },
            "kind": "SpanKind.INTERNAL",
            "parent_id": "0x1111111111111111",
            "start_time": "2024-01-01T10:00:00.500Z",
            "end_time": "2024-01-01T10:00:02.000Z",
            "status": {"status_code": "OK"},
            "attributes": {
                "span.type": "retrieval",
                "workflow.name": "test_workflow",
                "entity.1.type": "vectorstore.chroma",
                "entity.1.name": "knowledge_base"
            },
            "events": [
                {
                    "name": "data.input",
                    "timestamp": "2024-01-01T10:00:00.500Z",
                    "attributes": {"query": "What is machine learning?"}
                },
                {
                    "name": "data.output", 
                    "timestamp": "2024-01-01T10:00:02.000Z",
                    "attributes": {"results": ["ML is...", "AI involves..."]}
                }
            ],
            "links": []
        },
        {
            "name": "openai.chat.completions.completions.Completions",
            "context": {
                "trace_id": "0x1234567890abcdef",
                "span_id": "0x3333333333333333",
                "trace_state": "[]"
            },
            "kind": "SpanKind.INTERNAL", 
            "parent_id": "0x1111111111111111",
            "start_time": "2024-01-01T10:00:02.100Z",
            "end_time": "2024-01-01T10:00:04.500Z",
            "status": {"status_code": "OK"},
            "attributes": {
                "span.type": "inference",
                "workflow.name": "test_workflow",
                "entity.1.type": "inference.openai",
                "entity.1.provider_name": "api.openai.com",
                "entity.2.name": "gpt-4",
                "entity.2.type": "model.llm.gpt-4"
            },
            "events": [
                {
                    "name": "data.input",
                    "timestamp": "2024-01-01T10:00:02.100Z",
                    "attributes": {
                        "input": ["Context: ML is...", "Question: What is machine learning?"]
                    }
                },
                {
                    "name": "data.output",
                    "timestamp": "2024-01-01T10:00:04.500Z", 
                    "attributes": {
                        "response": "Machine learning is a subset of artificial intelligence..."
                    }
                }
            ],
            "links": []
        }
    ]


class TestTraceVisualization:
    """Test suite for trace visualization functionality."""
    
    def test_timeline_event_creation(self):
        """Test creation of timeline events from span data."""
        traces = create_sample_traces()
        gantt = TraceGanttChart(traces)
        events = gantt.parse_spans()
        
        assert len(events) == 3
        
        # Check root event
        root_events = [e for e in events if e.parent_id is None]
        assert len(root_events) == 1
        
        root = root_events[0]
        assert root.name == "workflow"
        assert root.span_type == "workflow"
        assert root.workflow_name == "test_workflow"
        assert len(root.children) == 2
        
    def test_hierarchical_structure(self):
        """Test that parent-child relationships are built correctly."""
        traces = create_sample_traces()
        gantt = TraceGanttChart(traces)
        events = gantt.parse_spans()
        
        # Find workflow span
        workflow = next(e for e in events if e.name == "workflow")
        
        # Should have 2 children
        assert len(workflow.children) == 2
        
        child_names = {child.name for child in workflow.children}
        assert "retrieval.vector_store" in child_names
        assert "openai.chat.completions.completions.Completions" in child_names
        
        # Children should have correct levels
        for child in workflow.children:
            assert child.level == 1
            
    def test_gantt_text_generation(self):
        """Test generation of text-based Gantt chart."""
        traces = create_sample_traces()
        gantt = TraceGanttChart(traces)
        text_output = gantt.generate_gantt_text()
        
        assert "=== Trace Gantt Chart ===" in text_output
        assert "workflow" in text_output
        assert "retrieval" in text_output
        assert "inference" in text_output
        assert "5000.0ms" in text_output  # Total duration
        
    def test_mermaid_generation(self):
        """Test generation of Mermaid diagram."""
        traces = create_sample_traces()
        gantt = TraceGanttChart(traces)
        mermaid_output = gantt.generate_mermaid_gantt()
        
        assert mermaid_output.startswith("gantt")
        assert "title Trace Execution Flow" in mermaid_output
        assert "test_workflow" in mermaid_output
        
    def test_critical_path_analysis(self):
        """Test critical path calculation."""
        traces = create_sample_traces()
        gantt = TraceGanttChart(traces)
        gantt.parse_spans()
        
        critical_path = gantt.get_critical_path()
        
        # Should include the workflow as the root
        assert len(critical_path) > 0
        assert critical_path[0].name == "workflow"
        
    def test_flow_pattern_validation(self):
        """Test flow pattern validation."""
        traces = create_sample_traces()
        gantt = TraceGanttChart(traces)
        gantt.parse_spans()
        
        validator = FlowValidator(gantt)
        
        # Test valid pattern
        result = validator.validate_pattern("workflow -> retrieval")
        assert result["valid"] is True
        assert len(result["matches"]) > 0
        
        # Test invalid pattern
        result = validator.validate_pattern("inference -> workflow")
        assert result["valid"] is False
        assert len(result["violations"]) > 0
        
    def test_pattern_suggestions(self):
        """Test automatic pattern suggestion."""
        traces = create_sample_traces() 
        gantt = TraceGanttChart(traces)
        gantt.parse_spans()
        
        validator = FlowValidator(gantt)
        suggestions = validator.suggest_patterns()
        
        # Should find some patterns
        assert len(suggestions) > 0
        
    def test_flow_validation_report(self):
        """Test generation of flow validation report."""
        traces = create_sample_traces()
        gantt = TraceGanttChart(traces)
        gantt.parse_spans()
        
        validator = FlowValidator(gantt)
        patterns = ["workflow -> retrieval", "retrieval -> inference"]
        
        report = validator.generate_flow_report(patterns)
        
        assert "=== Flow Validation Report ===" in report
        assert "workflow -> retrieval" in report
        assert "✓" in report or "✗" in report
        
    def test_comprehensive_report(self):
        """Test generation of comprehensive visualization report."""
        traces = create_sample_traces()
        gantt = TraceGanttChart(traces)
        
        patterns = ["workflow -> retrieval -> inference"]
        report = generate_visualization_report(gantt, patterns)
        
        assert "MONOCLE TRACE VISUALIZATION REPORT" in report
        assert "TIMELINE VISUALIZATION" in report
        assert "CRITICAL PATH ANALYSIS" in report
        assert "FLOW VALIDATION" in report
        assert "MERMAID GANTT DIAGRAM" in report
        
    def test_json_export(self):
        """Test JSON export functionality."""
        traces = create_sample_traces()
        gantt = TraceGanttChart(traces)
        gantt.parse_spans()
        
        json_output = gantt.to_json()
        data = json.loads(json_output)
        
        assert "timeline_events" in data
        assert "root_events" in data
        assert "critical_path" in data
        assert len(data["timeline_events"]) == 3
        assert len(data["root_events"]) == 1
        
    def test_timing_calculations(self):
        """Test timing and duration calculations."""
        traces = create_sample_traces()
        gantt = TraceGanttChart(traces)
        events = gantt.parse_spans()
        
        workflow_event = next(e for e in events if e.name == "workflow")
        
        # Workflow should be 5000ms (5 seconds)
        assert workflow_event.duration_ms == 5000.0
        
        retrieval_event = next(e for e in events if "retrieval" in e.name)
        
        # Retrieval should be 1500ms (1.5 seconds)
        assert retrieval_event.duration_ms == 1500.0
        
    def test_parallel_execution_validation(self):
        """Test validation of parallel execution patterns."""
        # Create traces with overlapping time periods
        parallel_traces = [
            {
                "name": "workflow", 
                "context": {"trace_id": "0x123", "span_id": "0x111"},
                "parent_id": None,
                "start_time": "2024-01-01T10:00:00.000Z",
                "end_time": "2024-01-01T10:00:06.000Z",
                "attributes": {"span.type": "workflow"}
            },
            {
                "name": "inference_a",
                "context": {"trace_id": "0x123", "span_id": "0x222"},
                "parent_id": "0x111",
                "start_time": "2024-01-01T10:00:01.000Z", 
                "end_time": "2024-01-01T10:00:03.000Z",
                "attributes": {"span.type": "inference"}
            },
            {
                "name": "inference_b",
                "context": {"trace_id": "0x123", "span_id": "0x333"},
                "parent_id": "0x111",
                "start_time": "2024-01-01T10:00:02.000Z",
                "end_time": "2024-01-01T10:00:04.000Z", 
                "attributes": {"span.type": "inference"}
            }
        ]
        
        gantt = TraceGanttChart(parallel_traces)
        gantt.parse_spans()
        
        validator = FlowValidator(gantt)
        result = validator.validate_pattern("inference_a || inference_b")
        
        # Should detect parallel execution
        assert result["valid"] is True


def test_visualization_demo():
    """Demonstrate the complete visualization functionality."""
    print("\\n" + "="*60)
    print("MONOCLE TRACE VISUALIZATION DEMO")
    print("="*60)
    
    # Create sample traces
    traces = create_sample_traces()
    
    # Generate comprehensive report
    gantt = TraceGanttChart(traces)
    patterns = [
        "workflow -> retrieval",
        "retrieval -> inference", 
        "workflow -> inference"
    ]
    
    report = generate_visualization_report(gantt, patterns)
    print(report)
    
    return True


if __name__ == "__main__":
    test_visualization_demo()