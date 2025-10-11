"""
Command-line interface for Monocle trace visualization.

This module provides a simple CLI tool for generating Gantt charts and 
flow validation reports from trace data files.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from ..assertions.flow_validator import FlowPattern, FlowValidator
from .examples import generate_visualization_report, get_example_patterns
from .gantt_chart import TraceGanttChart


def load_traces_from_file(file_path: str) -> List[dict]:
    """Load trace data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Handle different file formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'spans' in data:
            return data['spans']
        elif isinstance(data, dict) and 'traces' in data:
            return data['traces']
        else:
            print(f"Warning: Unexpected JSON structure in {file_path}")
            return [data] if isinstance(data, dict) else []
            
    except Exception as e:
        print(f"Error loading traces from {file_path}: {e}")
        return []


def generate_gantt_cli(trace_file: str, output_file: Optional[str] = None,
                      patterns: Optional[List[str]] = None,
                      format_type: str = "text") -> bool:
    """
    Generate Gantt chart visualization from trace file.
    
    Args:
        trace_file: Path to JSON file containing trace data
        output_file: Optional output file (defaults to stdout)
        patterns: Optional list of flow patterns to validate
        format_type: Output format ("text", "json", "mermaid")
        
    Returns:
        True if successful, False otherwise
    """
    traces = load_traces_from_file(trace_file)
    if not traces:
        print(f"No traces found in {trace_file}")
        return False
        
    try:
        gantt = TraceGanttChart(traces)
        gantt.parse_spans()
        
        if format_type == "text":
            if patterns:
                content = generate_visualization_report(gantt, patterns)
            else:
                content = gantt.generate_gantt_text()
        elif format_type == "json":
            content = gantt.to_json()
        elif format_type == "mermaid":
            content = gantt.generate_mermaid_gantt()
        else:
            print(f"Unknown format: {format_type}")
            return False
            
        if output_file:
            with open(output_file, 'w') as f:
                f.write(content)
            print(f"Gantt chart written to {output_file}")
        else:
            print(content)
            
        return True
        
    except Exception as e:
        print(f"Error generating Gantt chart: {e}")
        return False


def validate_flows_cli(trace_file: str, patterns: List[str],
                      output_file: Optional[str] = None) -> bool:
    """
    Validate flow patterns against trace data.
    
    Args:
        trace_file: Path to JSON file containing trace data  
        patterns: List of flow pattern strings to validate
        output_file: Optional output file (defaults to stdout)
        
    Returns:
        True if successful, False otherwise
    """
    traces = load_traces_from_file(trace_file)
    if not traces:
        print(f"No traces found in {trace_file}")
        return False
        
    try:
        gantt = TraceGanttChart(traces)
        gantt.parse_spans()
        
        validator = FlowValidator(gantt)
        
        flow_patterns = []
        for i, pattern_str in enumerate(patterns):
            flow_patterns.append(FlowPattern(f"pattern_{i+1}", pattern_str))
            
        content = validator.generate_flow_report(flow_patterns)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(content)
            print(f"Flow validation report written to {output_file}")
        else:
            print(content)
            
        return True
        
    except Exception as e:
        print(f"Error validating flows: {e}")
        return False


def list_example_patterns():
    """List available example patterns."""
    patterns = get_example_patterns()
    
    print("Available example patterns:")
    print("-" * 40)
    
    for name, pattern in patterns.items():
        print(f"{name}:")
        print(f"  Pattern: {pattern.pattern}")
        print(f"  Description: {pattern.description}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Monocle trace visualization and flow validation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text Gantt chart
  python -m monocle_tfwk.visualization.cli traces.json
  
  # Generate with flow validation
  python -m monocle_tfwk.visualization.cli traces.json -p "workflow -> inference"
  
  # Export as Mermaid diagram
  python -m monocle_tfwk.visualization.cli traces.json -f mermaid -o chart.mmd
  
  # Validate multiple patterns
  python -m monocle_tfwk.visualization.cli traces.json \\
    -p "retrieval -> inference" "workflow -> *" -o report.txt
    
  # List example patterns
  python -m monocle_tfwk.visualization.cli --examples
        """
    )
    
    parser.add_argument('trace_file', nargs='?',
                       help='JSON file containing trace data')
    
    parser.add_argument('-p', '--patterns', action='append',
                       help='Flow patterns to validate (can be used multiple times)')
    
    parser.add_argument('-f', '--format', choices=['text', 'json', 'mermaid'],
                       default='text', help='Output format (default: text)')
    
    parser.add_argument('-o', '--output', 
                       help='Output file (default: stdout)')
    
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run flow validation, skip Gantt chart')
    
    parser.add_argument('--examples', action='store_true',
                       help='List example patterns and exit')
    
    args = parser.parse_args()
    
    if args.examples:
        list_example_patterns()
        return
        
    if not args.trace_file:
        parser.print_help()
        print("\\nError: trace_file is required unless using --examples")
        sys.exit(1)
        
    if not Path(args.trace_file).exists():
        print(f"Error: Trace file {args.trace_file} not found")
        sys.exit(1)
        
    if args.validate_only:
        if not args.patterns:
            print("Error: --validate-only requires at least one pattern (-p)")
            sys.exit(1)
        success = validate_flows_cli(args.trace_file, args.patterns, args.output)
    else:
        success = generate_gantt_cli(args.trace_file, args.output, 
                                   args.patterns, args.format)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()