#!/usr/bin/env python3
"""
Trace Stitcher - Analyze and combine Monocle traces from the e-commerce example

This script reads all the trace files from the .monocle directory and creates
a comprehensive timeline of the shopping cart experience across all agents.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List


class TraceStitcher:
    def __init__(self, trace_directory: str):
        self.trace_directory = trace_directory
        self.traces = []
        self.events = []
        
    def load_traces(self) -> None:
        """Load all trace files from the directory."""
        trace_files = [f for f in os.listdir(self.trace_directory) if f.endswith('.json')]
        trace_files.sort()  # Sort by filename for consistent processing
        
        print(f"Found {len(trace_files)} trace files:")
        for trace_file in trace_files:
            print(f"  - {trace_file}")
            
        for trace_file in trace_files:
            file_path = os.path.join(self.trace_directory, trace_file)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Parse JSON array
                    trace_data = json.loads(content)
                    
                # Extract trace ID from filename
                trace_id_match = re.search(r'0x[a-f0-9]+', trace_file)
                trace_id = trace_id_match.group(0) if trace_id_match else "unknown"
                
                # Extract timestamp from filename
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})', trace_file)
                file_timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
                
                self.traces.append({
                    'file': trace_file,
                    'trace_id': trace_id,
                    'file_timestamp': file_timestamp,
                    'data': trace_data
                })
                print(f"    ✓ Loaded {len(trace_data)} spans")
                
            except Exception as e:
                print(f"    ✗ Error loading {trace_file}: {e}")
    
    def extract_events(self) -> None:
        """Extract key events from all traces and create a timeline."""
        for trace in self.traces:
            trace_id = trace['trace_id']
            file_timestamp = trace['file_timestamp']
            
            for span in trace['data']:
                # Extract basic span info
                span_name = span.get('name', 'unknown')
                start_time = span.get('start_time', '')
                end_time = span.get('end_time', '')
                attributes = span.get('attributes', {})
                events = span.get('events', [])
                
                # Parse timestamps
                start_dt = self._parse_timestamp(start_time)
                end_dt = self._parse_timestamp(end_time)
                
                # Extract agent and workflow info
                agent_name = attributes.get('entity.2.name', 'unknown')
                workflow_name = attributes.get('workflow.name', 'unknown')
                span_type = attributes.get('span.type', 'unknown')
                
                # Process different types of spans
                if 'generate_content' in span_name:
                    self._extract_llm_events(trace_id, span, start_dt, end_dt)
                elif 'FunctionTool.run_async' in span_name:
                    self._extract_tool_events(trace_id, span, start_dt, end_dt)
                elif 'Agent.run_async' in span_name:
                    self._extract_agent_events(trace_id, span, start_dt, end_dt)
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse ISO timestamp string to datetime object."""
        if not timestamp_str:
            return datetime.min
        try:
            # Remove Z suffix and parse
            clean_ts = timestamp_str.replace('Z', '+00:00')
            return datetime.fromisoformat(clean_ts)
        except Exception:
            return datetime.min
    
    def _extract_llm_events(self, trace_id: str, span: Dict, start_dt: datetime, end_dt: datetime) -> None:
        """Extract LLM inference events."""
        events = span.get('events', [])
        
        for event in events:
            if event.get('name') == 'data.input':
                input_data = event.get('attributes', {}).get('input', [])
                # Parse conversation from input
                conversation = self._parse_conversation(input_data)
                
                if conversation:
                    self.events.append({
                        'timestamp': start_dt,
                        'trace_id': trace_id,
                        'type': 'conversation',
                        'data': conversation,
                        'span_name': span.get('name', '')
                    })
            
            elif event.get('name') == 'metadata':
                metadata = event.get('attributes', {})
                if metadata:
                    self.events.append({
                        'timestamp': end_dt,
                        'trace_id': trace_id,
                        'type': 'llm_metrics',
                        'data': metadata,
                        'span_name': span.get('name', '')
                    })
    
    def _extract_tool_events(self, trace_id: str, span: Dict, start_dt: datetime, end_dt: datetime) -> None:
        """Extract tool invocation events."""
        attributes = span.get('attributes', {})
        tool_name = attributes.get('entity.1.name', 'unknown')
        
        events = span.get('events', [])
        input_data = None
        output_data = None
        
        for event in events:
            if event.get('name') == 'data.input':
                input_data = event.get('attributes', {}).get('input', [])
            elif event.get('name') == 'data.output':
                output_data = event.get('attributes', {}).get('response', '')
        
        self.events.append({
            'timestamp': start_dt,
            'trace_id': trace_id,
            'type': 'tool_call',
            'data': {
                'tool_name': tool_name,
                'input': input_data,
                'output': output_data,
                'duration_ms': (end_dt - start_dt).total_seconds() * 1000
            },
            'span_name': span.get('name', '')
        })
    
    def _extract_agent_events(self, trace_id: str, span: Dict, start_dt: datetime, end_dt: datetime) -> None:
        """Extract agent-level events."""
        attributes = span.get('attributes', {})
        agent_name = attributes.get('entity.1.name', 'unknown')
        
        self.events.append({
            'timestamp': start_dt,
            'trace_id': trace_id,
            'type': 'agent_session',
            'data': {
                'agent_name': agent_name,
                'duration_ms': (end_dt - start_dt).total_seconds() * 1000
            },
            'span_name': span.get('name', '')
        })
    
    def _parse_conversation(self, input_data: List[str]) -> List[Dict]:
        """Parse conversation data from input array."""
        conversation = []
        
        for item in input_data:
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    for role, content in parsed.items():
                        if content:  # Skip null values
                            conversation.append({
                                'role': role,
                                'content': content
                            })
            except json.JSONDecodeError:
                continue
        
        return conversation
    
    def create_timeline(self) -> None:
        """Create a comprehensive timeline of all events."""
        # Sort events by timestamp
        self.events.sort(key=lambda x: x['timestamp'])
        
        print("\n" + "="*80)
        print("STITCHED E-COMMERCE TRACE TIMELINE")
        print("="*80)
        
        current_trace = None
        for i, event in enumerate(self.events):
            timestamp = event['timestamp'].strftime('%H:%M:%S.%f')[:-3]  # Show milliseconds
            trace_id = event['trace_id'][-8:]  # Show last 8 chars
            event_type = event['type']
            
            # Show trace boundaries
            if current_trace != event['trace_id']:
                current_trace = event['trace_id']
                print(f"\n📊 TRACE: {trace_id} ({timestamp})")
                print("-" * 60)
            
            if event_type == 'conversation':
                print(f"\n💬 CONVERSATION [{timestamp}]:")
                for msg in event['data']:
                    role = msg['role'].upper()
                    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    print(f"    {role}: {content}")
            
            elif event_type == 'tool_call':
                tool_data = event['data']
                print(f"\n🔧 TOOL CALL [{timestamp}]: {tool_data['tool_name']}")
                if tool_data['input']:
                    print(f"    Input: {tool_data['input']}")
                if tool_data['output']:
                    print(f"    Output: {tool_data['output']}")
                print(f"    Duration: {tool_data['duration_ms']:.1f}ms")
            
            elif event_type == 'llm_metrics':
                metrics = event['data']
                print(f"📈 LLM METRICS [{timestamp}]:")
                for key, value in metrics.items():
                    print(f"    {key}: {value}")
            
            elif event_type == 'agent_session':
                agent_data = event['data']
                print(f"\n🤖 AGENT SESSION [{timestamp}]: {agent_data['agent_name']}")
                print(f"    Duration: {agent_data['duration_ms']:.1f}ms")
    
    def generate_summary(self) -> None:
        """Generate a summary of the complete trace analysis."""
        print(f"\n" + "="*80)
        print("TRACE SUMMARY")
        print("="*80)
        
        # Count events by type
        event_counts = {}
        for event in self.events:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        print(f"Total traces processed: {len(self.traces)}")
        print(f"Total events extracted: {len(self.events)}")
        print("\nEvent breakdown:")
        for event_type, count in sorted(event_counts.items()):
            print(f"  {event_type}: {count}")
        
        # Extract tool calls
        tool_calls = [e for e in self.events if e['type'] == 'tool_call']
        if tool_calls:
            print(f"\nTool calls made:")
            tool_names = {}
            for call in tool_calls:
                tool_name = call['data']['tool_name']
                tool_names[tool_name] = tool_names.get(tool_name, 0) + 1
            
            for tool_name, count in sorted(tool_names.items()):
                print(f"  {tool_name}: {count} calls")
        
        # Time span
        if self.events:
            start_time = min(e['timestamp'] for e in self.events)
            end_time = max(e['timestamp'] for e in self.events)
            duration = (end_time - start_time).total_seconds()
            print(f"\nTotal session duration: {duration:.2f} seconds")
            print(f"Start: {start_time.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"End: {end_time.strftime('%H:%M:%S.%f')[:-3]}")
    
    def run(self) -> None:
        """Run the complete trace stitching analysis."""
        print("🔍 Monocle Trace Stitcher")
        print("=" * 40)
        
        self.load_traces()
        self.extract_events()
        self.create_timeline()
        self.generate_summary()
        
        print(f"\n✅ Analysis complete! Found {len(self.events)} events across {len(self.traces)} traces.")


if __name__ == "__main__":
    # Use the .monocle directory
    trace_dir = "/Users/ravianne/projects/monocle/.monocle"
    
    if not os.path.exists(trace_dir):
        print(f"❌ Trace directory not found: {trace_dir}")
        exit(1)
    
    stitcher = TraceStitcher(trace_dir)
    stitcher.run()