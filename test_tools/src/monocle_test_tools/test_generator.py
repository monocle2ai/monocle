from typing import List, Set, Dict
from opentelemetry.sdk.trace import ReadableSpan


class TestGenerator:
    """Generates test code by analyzing trace spans."""
    
    def __init__(self, spans: List[ReadableSpan], trace_file: str = None):
        """Initialize with a list of spans to analyze."""
        self.spans = spans
        self.trace_file = trace_file
        self.agents: Set[str] = set()
        self.tools: Dict[str, str] = {}  # tool_name -> agent_name
        self.agent_outputs: Dict[str, List[str]] = {}  # agent_name -> outputs
        self.has_workflow = False
        
    @classmethod
    def from_json_file(cls, filepath: str):
        """Create generator from a trace JSON file."""
        from monocle_test_tools.span_loader import JSONSpanLoader
        spans = JSONSpanLoader.from_json(filepath)
        return cls(spans, trace_file=filepath)
    
    @classmethod
    def from_okahu(cls, trace_id: str, workflow_name: str):
        """Create generator from an Okahu trace."""
        from monocle_test_tools.span_loader import OkahuSpanLoader
        spans = OkahuSpanLoader.load_by_trace_id(
            trace_id=trace_id,
            workflow_name=workflow_name
        )
        return cls(spans, trace_file=None)
    
    def analyze(self):
        """Scan spans and extract agents, tools, and outputs."""
        for span in self.spans:
            span_type = span.attributes.get("span.type", "")
            
            if span_type == "agentic.invocation":
                name = span.attributes.get("entity.1.name", "")
                if name:
                    self.agents.add(name)
                    
                    events = getattr(span, 'events', [])
                    for event in events:
                        if event.name == "data.output":
                            content = event.attributes.get("response", "")
                            if content and len(content) > 10:  # Skip very short outputs
                                key_phrase = content[:80].strip()
                                if key_phrase:
                                    self.agent_outputs.setdefault(name, []).append(key_phrase)
            
            elif span_type == "agentic.tool.invocation":
                tool_name = span.attributes.get("entity.1.name", "")
                parent_agent = span.attributes.get("entity.2.name", "")
                if tool_name:
                    self.tools[tool_name] = parent_agent or ""
            
            elif span_type == "workflow":
                self.has_workflow = True
    
    def generate_test_code(self, test_name: str = "test_generated") -> str:
        """Generate Python test code with assertions."""
        
        self.analyze()
        
        code = [
            'import pytest',
            'from monocle_test_tools import TraceAssertion',
            'from monocle_test_tools.span_loader import JSONSpanLoader',
            '',
            '',
            f'def {test_name}(monocle_trace_asserter: TraceAssertion):',
            '    """Auto-generated test from trace analysis."""',
        ]
        
        # Add trace loading if we have a file
        if self.trace_file:
            code.extend([
                f'    spans = JSONSpanLoader.from_json("{self.trace_file}")',
                '    monocle_trace_asserter.validator.add_remote_spans(spans)',
                '',
            ])
        
        code.extend([
            '    asserter = monocle_trace_asserter',
            '',
        ])
        
        # Agent assertions with outputs
        if self.agents:
            code.append('    # Agent invocations with output checks')
            for agent in sorted(self.agents):
                outputs = self.agent_outputs.get(agent, [])
                if outputs:
                    output = outputs[0].replace('"', '\\"').replace('\n', ' ')
                    code.append(f'    asserter.called_agent("{agent}").contains_output("{output}")')
                else:
                    code.append(f'    asserter.called_agent("{agent}")')
            code.append('')
        
        # Tool assertions
        if self.tools:
            code.append('    # Tool invocations')
            for tool_name, agent_name in sorted(self.tools.items()):
                if agent_name:
                    code.append(f'    asserter.called_tool("{tool_name}", "{agent_name}")')
                else:
                    code.append(f'    asserter.called_tool("{tool_name}")')
            code.append('')
        
        return '\n'.join(code)
    
    def write_to_file(self, filepath: str):
        """Write generated test code to a file."""
        code = self.generate_test_code()
        with open(filepath, 'w') as f:
            f.write(code)
        print(f"Test written to: {filepath}")
