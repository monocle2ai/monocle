import math
from typing import List, Optional, Set, Dict
from opentelemetry.sdk.trace import ReadableSpan


# Allowed values for the ``trace_source`` argument. When set, only the loader
# code for that specific source is generated.
SUPPORTED_TRACE_SOURCES = ("file", "okahu")


class TestGenerator:
    """Generates test code by analyzing trace spans."""

    def __init__(self, spans: List[ReadableSpan], trace_file: str = None,
                 trace_source: Optional[str] = None, trace_id: Optional[str] = None,
                 workflow_name: Optional[str] = None):
        """Initialize with a list of spans to analyze.

        Args:
            spans: Spans to analyze.
            trace_file: Path to the trace file the spans were loaded from (if any).
            trace_source: Optional loader to generate code for. One of
                ``"file"`` or ``"okahu"``. When ``None`` (default), loader code
                for all supported sources is emitted.
            trace_id: Trace id, used when generating "okahu" loader code.
            workflow_name: Okahu workflow name, used when generating "okahu" loader code.
        """
        if trace_source is not None and trace_source not in SUPPORTED_TRACE_SOURCES:
            raise ValueError(
                f"Unsupported trace_source: '{trace_source}'. "
                f"Supported values: {', '.join(SUPPORTED_TRACE_SOURCES)}."
            )
        self.spans = spans
        self.trace_file = trace_file
        self.trace_source = trace_source
        self.trace_id = trace_id
        self.workflow_name = workflow_name
        self.agents: Set[str] = set()
        self.tools: Dict[str, str] = {}  # tool_name -> agent_name
        self.agent_outputs: Dict[str, List[str]] = {}  # agent_name -> outputs
        self.agent_inputs: Dict[str, List[str]] = {}  # agent_name -> inputs
        self.tool_inputs: Dict[str, str] = {}  # tool_name -> first input snippet
        self.tool_outputs: Dict[str, str] = {}  # tool_name -> first output snippet
        self.span_attributes: Dict[str, Dict[str, str]] = {}  # span_type -> {attr_key: attr_value}
        self.has_workflow = False
        self.total_tokens = 0  # total tokens across inference spans in the turn
        self.turn_duration = 0.0  # max agentic.turn duration in seconds

    @classmethod
    def from_json_file(cls, filepath: str, trace_source: Optional[str] = None):
        """Create generator from a trace JSON file."""
        from monocle_test_tools.span_loader import JSONSpanLoader
        spans = JSONSpanLoader.from_json(filepath)
        return cls(spans, trace_file=filepath, trace_source=trace_source)

    @classmethod
    def from_okahu(cls, trace_id: str, workflow_name: str, trace_source: Optional[str] = None):
        """Create generator from an Okahu trace."""
        from monocle_test_tools.span_loader import OkahuSpanLoader
        spans = OkahuSpanLoader.load_by_trace_id(
            trace_id=trace_id,
            workflow_name=workflow_name
        )
        return cls(spans, trace_file=None, trace_source=trace_source,
                   trace_id=trace_id, workflow_name=workflow_name)

    def analyze(self):
        """Scan spans and extract agents, tools, outputs, tokens and duration.

        Idempotent: resets accumulated state on each call so running it more than
        once (e.g. explicitly and again from generate_test_code) does not double
        token totals or duplicate outputs.
        """
        self.agents = set()
        self.tools = {}
        self.agent_outputs = {}
        self.agent_inputs = {}
        self.tool_inputs = {}
        self.tool_outputs = {}
        self.span_attributes = {}
        self.has_workflow = False
        self.total_tokens = 0
        self.turn_duration = 0.0

        # Attributes worth surfacing as has_attribute() assertions per span type.
        _NOTABLE_ATTRS = ("entity.1.type", "workflow.name", "span.type")

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
                        elif event.name == "data.input":
                            content = event.attributes.get("input", "") or event.attributes.get("user_input", "")
                            if content and len(content) > 5:
                                self.agent_inputs.setdefault(name, []).append(content[:80].strip())

            elif span_type == "agentic.tool.invocation":
                tool_name = span.attributes.get("entity.1.name", "")
                parent_agent = span.attributes.get("entity.2.name", "")
                if tool_name:
                    self.tools[tool_name] = parent_agent or ""
                    events = getattr(span, 'events', [])
                    for event in events:
                        if event.name == "data.input" and tool_name not in self.tool_inputs:
                            content = event.attributes.get("input", "") or event.attributes.get("user_input", "")
                            if content:
                                self.tool_inputs[tool_name] = content[:80].strip()
                        elif event.name == "data.output" and tool_name not in self.tool_outputs:
                            content = event.attributes.get("response", "") or event.attributes.get("output", "")
                            if content:
                                self.tool_outputs[tool_name] = content[:80].strip()

            elif span_type == "workflow":
                self.has_workflow = True

            # Collect notable span attributes per span type for has_attribute() assertions.
            if span_type:
                attrs = self.span_attributes.setdefault(span_type, {})
                for key in _NOTABLE_ATTRS:
                    val = span.attributes.get(key)
                    if val and key not in attrs:
                        attrs[key] = str(val)

            # Accumulate total tokens across inference spans in the turn.
            if span_type in ("inference", "inference.framework"):
                for event in getattr(span, 'events', []):
                    if event.name == "metadata":
                        self.total_tokens += event.attributes.get("total_tokens", 0) or 0

            # Track the duration of the agentic turn.
            if span_type == "agentic.turn" and span.start_time and span.end_time:
                duration = (span.end_time - span.start_time) / 1e9
                self.turn_duration = max(self.turn_duration, duration)

    def _generate_loading_lines(self) -> List[str]:
        """Generate the trace-loading section using the with_trace_source API.

        Honors ``self.trace_source``: when set to "file" or "okahu", only the
        loader for that source is emitted (as active code). When ``None``, all
        supported loaders are shown with the file loader active (if a trace file
        is known) and the rest commented out.
        """
        file_line = (
            f'    monocle_trace_asserter.with_trace_source("file", trace_path="{self.trace_file}")'
            if self.trace_file
            else '    monocle_trace_asserter.with_trace_source("file", trace_path="path/to/trace.json")'
        )
        okahu_id = self.trace_id or "TRACE_ID"
        okahu_workflow = self.workflow_name or "WORKFLOW_NAME"
        okahu_line = (
            f'    monocle_trace_asserter.with_trace_source("okahu", '
            f'id="{okahu_id}", workflow_name="{okahu_workflow}")'
        )

        if self.trace_source == "file":
            return ['    # Load traces from a local trace file', file_line]

        if self.trace_source == "okahu":
            return ['    # Load traces from Okahu', okahu_line]

        # Default: emit all options, file loader active when available.
        lines = ['    # Option 1: Load from a local trace file']
        if self.trace_file:
            lines.append(file_line)
        else:
            lines.append('    # ' + file_line.strip())
        lines.extend([
            '',
            '    # Option 2: Load from Okahu',
            '    # ' + okahu_line.strip(),
            '',
            '    # Option 3: Run agent directly',
            '    # from your_module import your_agent',
            '    # await monocle_trace_asserter.run_agent_async(your_agent, "framework_name", "user input")',
        ])
        return lines

    def generate_test_code(self, test_name: str = "test_generated") -> str:
        """Generate Python test code with assertions."""
        
        self.analyze()
        
        code = [
            'import pytest',
            'from monocle_test_tools import TraceAssertion',
            '',
            '',
            f'def {test_name}(monocle_trace_asserter: TraceAssertion):',
            '    """Auto-generated test from trace analysis."""',
            '',
        ]

        # Trace loading via the with_trace_source API.
        code.extend(self._generate_loading_lines())

        code.extend([
            '',
            '    asserter = monocle_trace_asserter',
            '',
        ])
        
        # Agent assertions with inputs, outputs and notable attributes
        if self.agents:
            code.append('    # Agent invocations with output checks')
            for agent in sorted(self.agents):
                outputs = self.agent_outputs.get(agent, [])
                inputs = self.agent_inputs.get(agent, [])
                chain = f'    asserter.called_agent("{agent}")'
                if outputs:
                    output = outputs[0].replace('"', '\\"').replace('\n', ' ')
                    chain += f'.contains_output("{output}")'
                if inputs:
                    inp = inputs[0].replace('"', '\\"').replace('\n', ' ')
                    chain += f'.has_input("{inp}")'
                code.append(chain)
            code.append('')

        # Notable span-type attributes as has_attribute() assertions
        _ASSERTION_SPAN_TYPES = ("agentic.invocation", "agentic.tool.invocation", "workflow")
        attr_lines = []
        for stype in _ASSERTION_SPAN_TYPES:
            attrs = self.span_attributes.get(stype, {})
            for key, val in sorted(attrs.items()):
                if key == "span.type":
                    continue  # redundant with called_agent / called_tool
                escaped_val = val.replace('"', '\\"')
                attr_lines.append(f'    asserter.has_attribute("{key}", "{escaped_val}")')
        if attr_lines:
            code.append('    # Span attribute assertions')
            code.extend(attr_lines)
            code.append('')

        # Tool assertions with optional input/output snippets
        if self.tools:
            code.append('    # Tool invocations')
            for tool_name, agent_name in sorted(self.tools.items()):
                chain = f'    asserter.called_tool("{tool_name}"'
                if agent_name:
                    chain += f', "{agent_name}"'
                chain += ')'
                if tool_name in self.tool_inputs:
                    inp = self.tool_inputs[tool_name].replace('"', '\\"').replace('\n', ' ')
                    chain += f'.has_input("{inp}")'
                if tool_name in self.tool_outputs:
                    out = self.tool_outputs[tool_name].replace('"', '\\"').replace('\n', ' ')
                    chain += f'.has_output("{out}")'
                code.append(chain)
            code.append('')

        # Cost check: total tokens in the turn
        if self.total_tokens > 0:
            code.append('    # Cost check: total tokens in the turn (derived from trace; adjust as needed)')
            code.append(f'    asserter.under_token_limit({self.total_tokens})')
            code.append('')

        # Performance check: duration of the turn
        if self.turn_duration > 0:
            # Round the limit up so the generated test passes against the source trace.
            duration_limit = math.ceil(self.turn_duration * 10) / 10
            code.append('    # Performance check: duration of the turn (derived from trace; adjust as needed)')
            code.append(f'    asserter.under_duration({duration_limit}, units="seconds", span_type="agent_turn")')
            code.append('')

        return '\n'.join(code)
    
    def write_to_file(self, filepath: str):
        """Write generated test code to a file."""
        code = self.generate_test_code()
        with open(filepath, 'w') as f:
            f.write(code)
        print(f"Test written to: {filepath}")
