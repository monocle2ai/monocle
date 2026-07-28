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
                 workflow_name: Optional[str] = None, session_id: Optional[str] = None,
                 scope_name: Optional[str] = None, scope_id: Optional[str] = None):
        """Initialize with a list of spans to analyze.

        Args:
            spans: Spans to analyze.
            trace_file: Path to the trace file the spans were loaded from (if any).
            trace_source: Optional loader to generate code for. One of
                ``"file"`` or ``"okahu"``. When ``None`` (default), loader code
                for all supported sources is emitted.
            trace_id: Trace id, used when generating "okahu" loader code.
            workflow_name: Okahu workflow name, used when generating "okahu" loader code.
            session_id: Agentic session id. When set, the spans span multiple traces
                (a whole session) and the generated Okahu loader targets the session
                (``fact_name="session"``) instead of a single trace.
            scope_name / scope_id: A custom Okahu fact/scope (e.g. ``"test_id"``) and
                its value. When set, the spans span every trace under that fact and the
                generated Okahu loader targets it (``fact_name="scope"``). This is the
                general form; ``session_id`` is the ``agent_sessions`` special case.
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
        self.session_id = session_id
        self.scope_name = scope_name
        self.scope_id = scope_id
        self.agents: Set[str] = set()
        self.tools: Dict[str, str] = {}  # tool_name -> agent_name
        self.agent_outputs: Dict[str, List[str]] = {}  # agent_name -> outputs
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
        spans = OkahuSpanLoader.get_spans(
            workflow_name=workflow_name,
            trace_id=trace_id,
        )
        return cls(spans, trace_file=None, trace_source=trace_source,
                   trace_id=trace_id, workflow_name=workflow_name)

    @classmethod
    def from_okahu_scope(cls, scope_name: str, scope_id: str, workflow_name: str,
                         trace_source: Optional[str] = None):
        """Create generator from any Okahu fact/scope other than a single trace.

        Loads spans across every trace under ``scope_name == scope_id`` (e.g.
        ``"agent_sessions"``, ``"test_id"``, ``"conversations"``, or any custom
        scope), so assertions are extracted from the whole fact rather than one turn.

        The ``agent_sessions`` scope is emitted with the idiomatic
        ``fact_name="session"`` loader; every other scope uses ``fact_name="scope"``.
        """
        from monocle_test_tools.span_loader import OkahuSpanLoader
        spans = OkahuSpanLoader.load_by_scope(
            workflow_name=workflow_name,
            scope_name=scope_name,
            scope_id=scope_id,
        )
        if scope_name == OkahuSpanLoader.AGENT_SESSIONS_SCOPE:
            return cls(spans, trace_file=None, trace_source=trace_source,
                       workflow_name=workflow_name, session_id=scope_id)
        return cls(spans, trace_file=None, trace_source=trace_source,
                   workflow_name=workflow_name, scope_name=scope_name, scope_id=scope_id)

    @classmethod
    def from_okahu_session(cls, session_id: str, workflow_name: str,
                           trace_source: Optional[str] = None):
        """Create generator from an Okahu agentic session.

        A session spans multiple traces; this loads spans across every trace in the
        session so agent/tool/inference assertions are extracted from the whole
        session (loading a single trace id would miss most turns — the cause of
        session tests being generated with no assertions). Thin wrapper over
        ``from_okahu_scope`` with the ``agent_sessions`` scope.
        """
        from monocle_test_tools.span_loader import OkahuSpanLoader
        return cls.from_okahu_scope(
            scope_name=OkahuSpanLoader.AGENT_SESSIONS_SCOPE,
            scope_id=session_id,
            workflow_name=workflow_name,
            trace_source=trace_source,
        )

    def analyze(self):
        """Scan spans and extract agents, tools, outputs, tokens and duration.

        Idempotent: resets accumulated state on each call so running it more than
        once (e.g. explicitly and again from generate_test_code) does not double
        token totals or duplicate outputs.
        """
        self.agents = set()
        self.tools = {}
        self.agent_outputs = {}
        self.has_workflow = False
        self.total_tokens = 0
        self.turn_duration = 0.0
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
        okahu_workflow = self.workflow_name or "WORKFLOW_NAME"
        if self.session_id:
            # Session: load spans across every trace in the agent session.
            okahu_line = (
                f'    monocle_trace_asserter.with_trace_source("okahu", '
                f'id="{self.session_id}", fact_name="session", workflow_name="{okahu_workflow}")'
            )
            okahu_label = '    # Load traces from an Okahu session'
        elif self.scope_id:
            # Any other fact/scope: load spans across every trace under the scope.
            okahu_line = (
                f'    monocle_trace_asserter.with_trace_source("okahu", '
                f'id="{self.scope_id}", fact_name="scope", scope_name="{self.scope_name}", '
                f'workflow_name="{okahu_workflow}")'
            )
            okahu_label = f'    # Load traces from Okahu scope "{self.scope_name}"'
        else:
            okahu_id = self.trace_id or "TRACE_ID"
            okahu_line = (
                f'    monocle_trace_asserter.with_trace_source("okahu", '
                f'id="{okahu_id}", workflow_name="{okahu_workflow}")'
            )
            okahu_label = '    # Load traces from Okahu'

        if self.trace_source == "file":
            return ['    # Load traces from a local trace file', file_line]

        if self.trace_source == "okahu":
            return [okahu_label, okahu_line]

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
