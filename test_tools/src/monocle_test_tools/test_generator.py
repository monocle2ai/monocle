import math
from typing import List, Optional, Set, Dict
from opentelemetry.sdk.trace import ReadableSpan


# Allowed values for the ``trace_source`` argument. When set, only the loader
# code for that specific source is generated.
SUPPORTED_TRACE_SOURCES = ("file", "okahu")


SUPPORTED_EVAL_SOURCES = ("okahu",)


class TestGenerator:
    """Generates test code by analyzing trace spans.

    Emits a complete, runnable test file with assertions for agents, tools,
    outputs, total tokens, turn duration, and optional eval assertions.
    The Okahu cloud loader is pre-populated with trace id and workflow name.
    """

    def __init__(self, spans: List[ReadableSpan], trace_file: str = None,
                 trace_source: Optional[str] = None, trace_id: Optional[str] = None,
                 workflow_name: Optional[str] = None,
                 injected_evals: Optional[List[Dict]] = None,
                 eval_source: Optional[str] = "okahu"):
        """Initialize with a list of spans to analyze.

        Args:
            spans: Spans to analyze.
            trace_file: Path to the trace file the spans were loaded from (if any).
            trace_source: Optional loader to generate code for. One of
                ``"file"`` or ``"okahu"``. When ``None`` (default), loader code
                for all supported sources is emitted.
            trace_id: Trace id, used when generating "okahu" loader code.
            workflow_name: Okahu workflow name, used when generating "okahu" loader code.
            injected_evals: Optional list of eval spec dicts passed directly from the CLI
                (``--eval`` flags). Each dict may contain ``criteria`` (built-in eval name)
                or ``template_path`` (custom JSON template), plus optional ``expected``,
                ``not_expected``, ``fact_name``, and ``eval_type`` (``"builtin"`` or
                ``"custom"``, auto-detected when absent).
            eval_source: Evaluator to use in generated ``with_evaluation()`` calls.
                Defaults to ``"okahu"``.
        """
        if trace_source is not None and trace_source not in SUPPORTED_TRACE_SOURCES:
            raise ValueError(
                f"Unsupported trace_source: '{trace_source}'. "
                f"Supported values: {', '.join(SUPPORTED_TRACE_SOURCES)}."
            )
        self.spans = spans
        self.trace_file = trace_file
        self.trace_source = trace_source
        self.eval_source = eval_source or "okahu"
        if self.eval_source not in SUPPORTED_EVAL_SOURCES:
            raise ValueError(
                f"Unsupported eval_source: '{self.eval_source}'. "
                f"Supported values: {', '.join(SUPPORTED_EVAL_SOURCES)}."
            )
        self.trace_id = trace_id
        self.workflow_name = workflow_name
        # Evals injected directly via CLI --eval flags (after type detection/normalisation).
        self._injected_evals: List[Dict] = [
            self._normalise_injected_eval(ev) for ev in (injected_evals or [])
            if ev  # skip empty
        ]
        self.agents: Set[str] = set()
        self.tools: Dict[str, str] = {}  # tool_name -> agent_name
        self.agent_outputs: Dict[str, List[str]] = {}  # agent_name -> outputs
        self.has_workflow = False
        self.total_tokens = 0  # total tokens across inference spans in the turn
        self.turn_duration = 0.0  # max agentic.turn duration in seconds
        self.evals: List[Dict[str, object]] = []  # eval specs supplied as parameters (deduped)

    @classmethod
    def from_json_file(cls, filepath: str, trace_source: Optional[str] = None,
                       injected_evals: Optional[List[Dict]] = None,
                       eval_source: Optional[str] = "okahu"):
        """Create generator from a trace JSON file."""
        from monocle_test_tools.span_loader import JSONSpanLoader
        spans = JSONSpanLoader.from_json(filepath)
        return cls(spans, trace_file=filepath, trace_source=trace_source,
                   injected_evals=injected_evals, eval_source=eval_source)

    @classmethod
    def from_okahu(cls, trace_id: str, workflow_name: str, trace_source: Optional[str] = None,
                   injected_evals: Optional[List[Dict]] = None,
                   eval_source: Optional[str] = "okahu"):
        """Create generator from an Okahu trace."""
        from monocle_test_tools.span_loader import OkahuSpanLoader
        spans = OkahuSpanLoader.get_spans(
            trace_id=trace_id,
            workflow_name=workflow_name,
        )
        return cls(spans, trace_file=None, trace_source=trace_source,
                   trace_id=trace_id, workflow_name=workflow_name,
                   injected_evals=injected_evals, eval_source=eval_source)

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
        self.evals = []
        for span in self.spans:
            span_type = span.attributes.get("span.type", "")

            # Derive trace_id / workflow_name from the span; explicit values take precedence.
            if not self.workflow_name:
                wf = span.attributes.get("workflow.name")
                if wf:
                    self.workflow_name = wf
            if not self.trace_id:
                self.trace_id = self._get_trace_id(span)

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

        # Evals are supplied as parameters (CLI --eval flags / injected_evals);
        # de-duplicate them, preserving first-seen order.
        deduped: List[Dict[str, object]] = []
        seen: Set[tuple] = set()
        for ev in self._injected_evals:
            key = (ev.get("criteria"), ev.get("template_path"),
                   repr(ev.get("expected")), repr(ev.get("not_expected")), ev.get("fact_name"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ev)
        self.evals = deduped

    @staticmethod
    def _get_trace_id(span) -> Optional[str]:
        """Best-effort 32-hex trace id from a span's context."""
        try:
            ctx = span.get_span_context()
            tid = getattr(ctx, "trace_id", None)
            if isinstance(tid, int):
                return format(tid, "032x")
            if tid:
                return str(tid).removeprefix("0x")
        except Exception:
            pass
        return None

    @staticmethod
    def _detect_eval_type(name_or_path: str, eval_source: str = "okahu") -> str:
        """Classify eval input as ``"builtin"`` or ``"custom"`` using the eval
        source's own rules (via its ``BaseEval.classify_eval_input``)."""
        from monocle_test_tools.evals.eval_manager import get_evaluator_class
        evaluator_cls = get_evaluator_class(eval_source)
        eval_type, _ = evaluator_cls.classify_eval_input(name_or_path)
        return eval_type

    def _normalise_injected_eval(self, raw: Dict) -> Dict:
        """Normalise a raw injected eval dict (from CLI --eval parsing / injected_evals)
        into the shape consumed by ``_eval_assertion_line``.

        Detects eval type via the configured ``eval_source`` when ``eval_type`` is
        absent and maps a custom value into ``template_path``. The evaluator for the
        emitted ``with_evaluation(...)`` is taken from ``eval_source`` at render time.
        """
        ev = dict(raw)

        # Honor an explicitly-passed eval_type; auto-detect only when it is
        # missing or not one of the two valid values.
        eval_type = ev.get("eval_type")
        if eval_type not in ("builtin", "custom"):
            name = ev.get("criteria") or ev.get("template_path") or ""
            eval_type = self._detect_eval_type(name, self.eval_source)
        ev["eval_type"] = eval_type

        # If a file path arrived in the ``criteria`` field (e.g. from a raw dict),
        # move it to template_path.
        if eval_type == "custom" and ev.get("criteria") and not ev.get("template_path"):
            ev["template_path"] = ev.pop("criteria")

        if not ev.get("fact_name"):
            ev["fact_name"] = "traces"

        return ev

    @staticmethod
    def _eval_literal(value) -> str:
        """Render a value as a Python source literal (string or list of strings)."""
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(TestGenerator._eval_literal(v) for v in value) + "]"
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{text}"'

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

        # For okahu source (or default), always emit all three loaders.
        # Active loader depends on trace_source: okahu → okahu line active, else file line active.
        okahu_active = self.trace_source == "okahu"

        lines = ['    # Option 1: Load from a local trace file']
        lines.append(('    # ' + file_line.strip()) if okahu_active else file_line)
        lines.extend([
            '',
            '    # Option 2: Load from Okahu',
            (okahu_line if okahu_active else ('    # ' + okahu_line.strip())),
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

        # Eval assertions (supplied as parameters; built-in and/or custom).
        if self.evals:
            code.append('    # Eval assertions (from eval parameters; require an eval service, e.g. Okahu)')
            for ev in self.evals:
                code.append('    ' + self._eval_assertion_line(ev))
            code.append('')

        return '\n'.join(code)

    def _eval_assertion_line(self, ev: Dict[str, object]) -> str:
        """Build a single active check_eval assertion line from a normalised eval spec.

        The spec's ``eval_type`` (``"builtin"``/``"custom"``) selects the call shape
        (``eval_name`` positional vs ``template_path`` keyword) and the inline
        ``# builtin eval`` / ``# custom eval`` comment shown alongside the assertion.
        """
        lit = self._eval_literal
        eval_type = ev.get("eval_type")  # "builtin" or "custom" (set by _normalise_injected_eval)

        # check_eval accepts an eval_name positional OR a template_path keyword (not both).
        if ev.get("template_path"):
            call_args = [f'template_path={lit(ev["template_path"])}']
        else:
            call_args = [lit(ev["criteria"])]
        if ev.get("expected") is not None:
            call_args.append(f'expected={lit(ev["expected"])}')
        if ev.get("not_expected") is not None:
            call_args.append(f'not_expected={lit(ev["not_expected"])}')
        if ev.get("fact_name"):
            call_args.append(f'fact_name={lit(ev["fact_name"])}')
        evaluator = self.eval_source
        args_str = ", ".join(call_args)
        line = f'asserter.with_evaluation("{evaluator}").check_eval({args_str})'

        # Append inline type comment for injected evals so developers know the type
        if eval_type in ("builtin", "custom"):
            line += f'  # {eval_type} eval'

        return line
    
    def write_to_file(self, filepath: str):
        """Write generated test code to a file."""
        code = self.generate_test_code()
        with open(filepath, 'w') as f:
            f.write(code)
        print(f"Test written to: {filepath}")
