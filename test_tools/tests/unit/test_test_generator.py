"""Unit tests for TestGenerator functionality."""

import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from monocle_test_tools.test_generator import TestGenerator


def test_from_json_file_basic():
    """Test loading generator from a trace file."""
    # Use an existing trace file
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    
    assert generator is not None
    assert generator.spans is not None
    assert len(generator.spans) > 0
    assert generator.trace_file == trace_path


def test_from_okahu_uses_span_loader_get_spans():
    """from_okahu should fetch spans via OkahuSpanLoader.get_spans."""
    fake_spans = []
    with patch("monocle_test_tools.span_loader.OkahuSpanLoader.get_spans", return_value=fake_spans) as mock_get_spans:
        generator = TestGenerator.from_okahu(trace_id="abc123", workflow_name="my_app")

    mock_get_spans.assert_called_once_with(trace_id="abc123", workflow_name="my_app")
    assert generator.spans == fake_spans
    assert generator.trace_id == "abc123"
    assert generator.workflow_name == "my_app"


def test_analyze_extracts_agents():
    """Test that analyze() correctly extracts agents from spans."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    generator.analyze()
    
    # Should find at least one agent
    assert len(generator.agents) > 0
    # Agents should be strings
    assert all(isinstance(agent, str) for agent in generator.agents)


def test_analyze_extracts_tools():
    """Test that analyze() correctly extracts tools from spans."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    generator.analyze()
    
    # Should find tools if trace has any
    if len(generator.tools) > 0:
        # Tools should map tool_name -> agent_name
        assert all(isinstance(name, str) for name in generator.tools.keys())


def test_generate_test_code_structure():
    """Test that generated code has correct structure."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    test_code = generator.generate_test_code(test_name="test_sample")
    
    # Check structure
    assert "import pytest" in test_code
    assert "from monocle_test_tools import TraceAssertion" in test_code
    assert "def test_sample(monocle_trace_asserter: TraceAssertion):" in test_code
    assert "asserter = monocle_trace_asserter" in test_code


def test_generate_includes_trace_loading():
    """Test that generated code includes trace loading when trace_file is set."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    test_code = generator.generate_test_code()

    # Should include trace loading code via the with_trace_source API
    assert "with_trace_source" in test_code
    assert trace_path in test_code


def test_generate_uses_with_trace_source_api():
    """Generated loading code should use the with_trace_source API, not the direct loader."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")

    generator = TestGenerator.from_json_file(trace_path)
    test_code = generator.generate_test_code()

    assert 'with_trace_source("file"' in test_code
    # The direct loader should no longer be used for loading.
    assert "JSONSpanLoader.from_json" not in test_code
    assert "add_remote_spans" not in test_code


def test_trace_source_file_only():
    """When trace_source='file', only the file loader is generated (as active code)."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")

    generator = TestGenerator.from_json_file(trace_path, trace_source="file")
    test_code = generator.generate_test_code()

    assert 'with_trace_source("file"' in test_code
    assert 'with_trace_source("okahu"' not in test_code
    # No Okahu scaffolding when a single source is requested.
    assert "Load from Okahu cloud" not in test_code


def test_trace_source_okahu_only():
    """When trace_source='okahu', the okahu loader is active and the file loader is commented out."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")

    generator = TestGenerator.from_json_file(trace_path, trace_source="okahu")
    test_code = generator.generate_test_code()

    assert 'monocle_trace_asserter.with_trace_source("okahu"' in test_code
    # File loader is present but commented out
    assert '# monocle_trace_asserter.with_trace_source("file"' in test_code


def test_invalid_trace_source_rejected():
    """An unsupported trace_source value should raise ValueError."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")

    with pytest.raises(ValueError):
        TestGenerator.from_json_file(trace_path, trace_source="invalid")


def test_includes_token_and_duration_checks():
    """Generated code should include under_token_limit and under_duration checks
    when the trace has token/turn data."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")

    generator = TestGenerator.from_json_file(trace_path)
    generator.analyze()
    test_code = generator.generate_test_code()

    if generator.total_tokens > 0:
        assert "under_token_limit" in test_code
    if generator.turn_duration > 0:
        assert "under_duration" in test_code
        assert 'span_type="agent_turn"' in test_code


def test_analyze_is_idempotent():
    """Running analyze() more than once must not double token totals or
    duplicate outputs (generate_test_code also calls analyze internally)."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")

    generator = TestGenerator.from_json_file(trace_path)
    generator.analyze()
    tokens_once = generator.total_tokens
    outputs_once = {a: list(v) for a, v in generator.agent_outputs.items()}

    generator.analyze()
    generator.analyze()

    assert generator.total_tokens == tokens_once
    assert generator.agent_outputs == outputs_once


def test_generate_includes_agent_assertions():
    """Test that generated code includes agent assertions."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    test_code = generator.generate_test_code()
    
    # Analyze to populate agents
    generator.analyze()
    
    # If there are agents, should have agent assertions
    if generator.agents:
        assert "called_agent" in test_code
        # Should mention at least one agent name
        assert any(agent in test_code for agent in generator.agents)


def test_generate_includes_tool_assertions():
    """Test that generated code includes tool assertions."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    test_code = generator.generate_test_code()
    
    # Analyze to populate tools
    generator.analyze()
    
    # If there are tools, should have tool assertions
    if generator.tools:
        assert "called_tool" in test_code


def test_write_to_file():
    """Test writing generated code to a file."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_path = f.name
    
    try:
        generator.write_to_file(temp_path)
        
        # Check file was created and has content
        assert Path(temp_path).exists()
        content = Path(temp_path).read_text()
        assert len(content) > 0
        assert "def test_generated" in content
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_custom_test_name():
    """Test that custom test name is used in generated code."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    test_code = generator.generate_test_code(test_name="test_my_custom_name")
    
    assert "def test_my_custom_name" in test_code


def test_output_checks_included():
    """Test that output checks are included for agents with outputs."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    generator.analyze()
    
    # If any agent has outputs, should include contains_output
    if any(generator.agent_outputs.values()):
        test_code = generator.generate_test_code()
        assert "contains_output" in test_code


def _event(name, attributes):
    return SimpleNamespace(name=name, attributes=attributes)


def _span(attributes, events=(), trace_id=0xABC123, start=None, end=None):
    return SimpleNamespace(
        attributes=attributes,
        events=list(events),
        start_time=start,
        end_time=end,
        get_span_context=lambda: SimpleNamespace(trace_id=trace_id),
    )


def test_okahu_loader_prepopulated_from_file_trace():
    """The Okahu cloud loader should be pre-populated with the workflow name and
    trace id derived from the trace, even when generating from a file."""
    spans = [_span({"span.type": "workflow", "workflow.name": "adk-travel-agent"})]
    generator = TestGenerator(spans, trace_file="my_trace.json")
    code = generator.generate_test_code()

    assert 'workflow_name="adk-travel-agent"' in code
    assert 'id="00000000000000000000000000abc123"' in code
    # Placeholders should no longer appear once real values are known.
    assert "WORKFLOW_NAME" not in code
    assert "TRACE_ID" not in code


# --- Eval injection (evals passed as parameters) ---------------------------------

def test_detect_eval_type():
    """Built-in names, .json/paths, and bare names are classified correctly."""
    assert TestGenerator._detect_eval_type("hallucination") == "builtin"   # known built-in
    assert TestGenerator._detect_eval_type("sentiment") == "builtin"
    assert TestGenerator._detect_eval_type("./my_eval.json") == "custom"    # path-like
    assert TestGenerator._detect_eval_type("templates/x.json") == "custom"
    assert TestGenerator._detect_eval_type("my_unknown_eval") == "builtin"  # bare name default


def test_injected_builtin_eval_emitted():
    """A built-in eval passed as a parameter emits check_eval(eval_name, expected=...)."""
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    evals = [{"criteria": "hallucination", "expected": "no_hallucination", "eval_type": "builtin"}]
    code = TestGenerator(spans, trace_file="t.json", injected_evals=evals).generate_test_code()

    assert 'with_evaluation("okahu")' in code
    assert 'check_eval("hallucination", expected="no_hallucination"' in code
    assert "# builtin eval" in code


def test_injected_custom_eval_emitted():
    """A custom template eval passed as a parameter emits check_eval(template_path=...)."""
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    evals = [{"template_path": "./my_eval.json", "expected": "pass", "eval_type": "custom"}]
    code = TestGenerator(spans, trace_file="t.json", injected_evals=evals).generate_test_code()

    assert 'check_eval(template_path="./my_eval.json", expected="pass"' in code
    assert "# custom eval" in code


def test_explicit_eval_type_is_honored_over_detection():
    """An explicitly-passed eval_type wins; a plain name marked custom moves to template_path."""
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    evals = [{"criteria": "my_template", "expected": "ok", "eval_type": "custom"}]
    gen = TestGenerator(spans, trace_file="t.json", injected_evals=evals)
    gen.analyze()

    assert gen.evals[0]["eval_type"] == "custom"
    assert gen.evals[0].get("template_path") == "my_template"
    assert "check_eval(template_path=\"my_template\"" in gen.generate_test_code()


def test_injected_evals_deduplicated():
    """The same eval passed twice as a parameter yields a single assertion."""
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    dup = {"criteria": "bias", "expected": "unbiased", "eval_type": "builtin"}
    gen = TestGenerator(spans, trace_file="t.json", injected_evals=[dict(dup), dict(dup)])
    gen.analyze()
    assert len(gen.evals) == 1


def test_from_okahu_passes_injected_evals():
    """from_okahu forwards injected evals through to the generator."""
    evals = [{"criteria": "sentiment", "expected": "positive"}]
    with patch("monocle_test_tools.span_loader.OkahuSpanLoader.get_spans", return_value=[]):
        gen = TestGenerator.from_okahu(trace_id="abc", workflow_name="wf", injected_evals=evals)
    code = gen.generate_test_code()
    assert 'check_eval("sentiment", expected="positive"' in code


# --- CLI --eval parsing ----------------------------------------------------------

def test_cli_parse_eval_spec_builtin_and_custom():
    from monocle_test_tools.generate_test import _parse_eval_spec

    b = _parse_eval_spec("hallucination=no_hallucination", "traces")
    assert b["eval_type"] == "builtin" and b["criteria"] == "hallucination"
    assert b["expected"] == "no_hallucination"

    c = _parse_eval_spec("./my_eval.json=pass", "traces")
    assert c["eval_type"] == "custom" and c["template_path"] == "./my_eval.json"


def test_cli_parse_eval_spec_explicit_type_prefix():
    from monocle_test_tools.generate_test import _parse_eval_spec

    # Force a plain name to be treated as a custom template path.
    c = _parse_eval_spec("custom:my_template=ok", "traces")
    assert c["eval_type"] == "custom" and c["template_path"] == "my_template"

    # Force a .json value to be treated as a built-in name (explicit override).
    b = _parse_eval_spec("builtin:weird.json=ok", "traces")
    assert b["eval_type"] == "builtin" and b["criteria"] == "weird.json"


def test_cli_parse_eval_spec_requires_expected():
    from monocle_test_tools.generate_test import _parse_eval_spec

    with pytest.raises(ValueError):
        _parse_eval_spec("hallucination", "traces")       # missing =EXPECTED
    with pytest.raises(ValueError):
        _parse_eval_spec("hallucination=", "traces")      # empty expected


# --- eval_source argument --------------------------------------------------------

def test_generated_with_evaluation_uses_eval_source():
    """The generated with_evaluation(...) call reflects the eval_source argument."""
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    evals = [{"criteria": "hallucination", "expected": "pass"}]
    code = TestGenerator(spans, trace_file="t.json", injected_evals=evals,
                         eval_source="okahu").generate_test_code()
    assert 'with_evaluation("okahu")' in code


def test_unsupported_eval_source_rejected():
    """An unsupported eval_source raises ValueError (like trace_source)."""
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    with pytest.raises(ValueError):
        TestGenerator(spans, trace_file="t.json", eval_source="not_a_real_evaluator")


def test_supported_eval_sources_matches_registry():
    """The local SUPPORTED_EVAL_SOURCES mirror must not drift from the registry."""
    from monocle_test_tools.test_generator import SUPPORTED_EVAL_SOURCES
    from monocle_test_tools.evals.eval_manager import get_supported_eval_sources
    assert set(SUPPORTED_EVAL_SOURCES) == set(get_supported_eval_sources())


def test_detect_eval_type_via_eval_source():
    """_detect_eval_type routes through the eval source's classify_eval_input."""
    assert TestGenerator._detect_eval_type("hallucination", "okahu") == "builtin"
    assert TestGenerator._detect_eval_type("./x.json", "okahu") == "custom"


def _session_spans():
    """Two traces (two turns) of one session: each has a workflow, an agent
    invocation (with output) and a tool invocation."""
    spans = []
    for i, (tid, agent, tool) in enumerate([
        (0xAAA1, "flight_agent", "book_flight"),
        (0xBBB2, "hotel_agent", "book_hotel"),
    ]):
        spans.append(_span({"span.type": "workflow", "workflow.name": "travel"}, trace_id=tid))
        spans.append(_span(
            {"span.type": "agentic.invocation", "entity.1.name": agent, "workflow.name": "travel"},
            events=[_event("data.output", {"response": f"{agent} finished the booking successfully."})],
            trace_id=tid,
        ))
        spans.append(_span(
            {"span.type": "agentic.tool.invocation", "entity.1.name": tool,
             "entity.2.name": agent, "workflow.name": "travel"},
            trace_id=tid,
        ))
    return spans


def test_from_okahu_uses_get_spans():
    """from_okahu must call the loader's get_spans (regression: it used a
    non-existent load_by_trace_id)."""
    with patch("monocle_test_tools.okahu_span_loader.OkahuSpanLoader.get_spans",
               return_value=[]) as mock_get:
        gen = TestGenerator.from_okahu(trace_id="abc123", workflow_name="wf")
    mock_get.assert_called_once_with(workflow_name="wf", trace_id="abc123")
    assert gen.trace_id == "abc123" and gen.workflow_name == "wf"


def test_from_okahu_session_loads_all_session_spans():
    """from_okahu_session delegates to from_okahu_scope with the agent_sessions scope
    and records the session id (idiomatic fact_name=session)."""
    session_spans = _session_spans()
    with patch("monocle_test_tools.okahu_span_loader.OkahuSpanLoader.load_by_scope",
               return_value=session_spans) as mock_load:
        gen = TestGenerator.from_okahu_session(session_id="sess_1", workflow_name="travel")
    mock_load.assert_called_once_with(
        workflow_name="travel", scope_name="agent_sessions", scope_id="sess_1")
    assert gen.session_id == "sess_1"
    assert gen.scope_name is None and gen.scope_id is None  # session uses the session form
    assert gen.spans == session_spans


def test_from_okahu_scope_generic_fact():
    """from_okahu_scope fetches any fact via load_by_scope and records scope_name/id."""
    scope_spans = _session_spans()
    with patch("monocle_test_tools.okahu_span_loader.OkahuSpanLoader.load_by_scope",
               return_value=scope_spans) as mock_load:
        gen = TestGenerator.from_okahu_scope(scope_name="test_id", scope_id="run_9",
                                             workflow_name="travel")
    mock_load.assert_called_once_with(
        workflow_name="travel", scope_name="test_id", scope_id="run_9")
    assert gen.scope_name == "test_id" and gen.scope_id == "run_9"
    assert gen.session_id is None
    # Assertions still extracted across all traces under the fact.
    assert 'called_agent("flight_agent")' in gen.generate_test_code()


def test_scope_loader_line_targets_the_fact():
    """A non-session fact emits fact_name="scope" with scope_name in the loader."""
    gen = TestGenerator(_session_spans(), scope_name="test_id", scope_id="run_9",
                        workflow_name="travel", trace_source="okahu")
    code = gen.generate_test_code()
    assert ('with_trace_source("okahu", id="run_9", fact_name="scope", '
            'scope_name="test_id", workflow_name="travel")') in code


def test_session_generates_assertions_across_turns():
    """The core fix: a session yields assertions for every turn's agents and tools,
    not an empty test."""
    gen = TestGenerator(_session_spans(), session_id="sess_1", workflow_name="travel")
    code = gen.generate_test_code()

    assert 'called_agent("flight_agent")' in code
    assert 'called_agent("hotel_agent")' in code
    assert 'called_tool("book_flight", "flight_agent")' in code
    assert 'called_tool("book_hotel", "hotel_agent")' in code


def test_session_loader_line_targets_the_session():
    """The generated Okahu loader targets the session (fact_name=session), not a trace."""
    gen = TestGenerator(_session_spans(), session_id="sess_1", workflow_name="travel",
                        trace_source="okahu")
    code = gen.generate_test_code()

    assert ('with_trace_source("okahu", id="sess_1", fact_name="session", '
            'workflow_name="travel")') in code
    assert "TRACE_ID" not in code


# --- Eval discovery: fact-level resolution + factory plumbing (Task 4) -----------

def test_resolve_discovery_fact_name_auto_match():
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]

    trace_gen = TestGenerator(spans, trace_file="t.json")
    assert trace_gen._resolve_discovery_fact_name() == "traces"

    sess_gen = TestGenerator(spans, session_id="s1", workflow_name="wf")
    assert sess_gen._resolve_discovery_fact_name() == "agentic_sessions"

    # A supported scope stays; an unsupported one falls back to traces.
    scope_ok = TestGenerator(spans, scope_name="conversations", scope_id="c1", workflow_name="wf")
    assert scope_ok._resolve_discovery_fact_name() == "conversations"
    scope_bad = TestGenerator(spans, scope_name="test_id", scope_id="r1", workflow_name="wf")
    assert scope_bad._resolve_discovery_fact_name() == "traces"


def test_discovery_fact_name_override_wins():
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    gen = TestGenerator(spans, session_id="s1", workflow_name="wf",
                        discovery_fact_name="traces")
    assert gen._resolve_discovery_fact_name() == "traces"


def test_factories_forward_discover_flag():
    with patch("monocle_test_tools.span_loader.OkahuSpanLoader.get_spans", return_value=[]):
        gen = TestGenerator.from_okahu(trace_id="abc", workflow_name="wf",
                                       discover_evals=False)
    assert gen.discover_evals is False


# --- Eval discovery: merge + emit (Task 5) ---------------------------------------

def _disc_spec(name, label, fact_id="abc123", fact_name="traces"):
    return {"criteria": name, "expected": label, "fact_name": fact_name,
            "eval_type": "builtin", "_discovered": True, "_discovered_fact_id": fact_id}


def test_discovered_eval_emitted_with_baseline_comment():
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.discover_fact_evals",
               return_value=([_disc_spec("correctness", "correct")], None)):
        code = TestGenerator(spans, trace_file="t.json").generate_test_code()
    assert 'check_eval("correctness", expected="correct", fact_name="traces")' in code
    assert "# discovered from fact abc123; adjust as needed" in code


def _disc_custom_spec(name, label, fact_id="abc123", fact_name="traces"):
    return {"criteria": name, "expected": label, "fact_name": fact_name,
            "eval_type": "custom", "_discovered": True,
            "_discovered_fact_id": fact_id, "_discovered_custom": True}


def test_discovered_custom_eval_is_commented_out_with_path_request():
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.discover_fact_evals",
               return_value=([_disc_custom_spec("hallucination", "major_hallucination")], None)):
        code = TestGenerator(spans, trace_file="t.json").generate_test_code()
    # No ACTIVE check_eval for the custom eval — the assertion line is commented out.
    assert "\n    asserter.with_evaluation" not in code  # no active eval line
    assert '# asserter.with_evaluation("okahu").check_eval(template_path="PATH/TO/your_custom_template.json"' in code
    assert 'expected="major_hallucination"' in code
    assert 'Custom eval "hallucination"' in code
    assert "Okahu does not store custom templates" in code


def test_discovered_builtin_and_custom_same_name_both_emitted():
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    discovered = [
        _disc_spec("hallucination", "major_hallucination"),          # builtin (active)
        _disc_custom_spec("hallucination", "major_hallucination"),   # custom (commented)
    ]
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.discover_fact_evals",
               return_value=(discovered, None)):
        gen = TestGenerator(spans, trace_file="t.json")
        code = gen.generate_test_code()
    # Both kept: an active builtin assertion AND a commented custom block.
    assert 'asserter.with_evaluation("okahu").check_eval("hallucination", expected="major_hallucination"' in code
    assert 'template_path="PATH/TO/your_custom_template.json"' in code


def test_no_evals_found_emits_comment():
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.discover_fact_evals",
               return_value=([], "No existing evals found on this fact")):
        code = TestGenerator(spans, trace_file="t.json").generate_test_code()
    assert "# No existing evals found on this fact" in code


def test_discovery_skipped_emits_comment():
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.discover_fact_evals",
               return_value=([], "eval discovery skipped: OKAHU_API_KEY not configured")):
        code = TestGenerator(spans, trace_file="t.json").generate_test_code()
    assert "# eval discovery skipped: OKAHU_API_KEY not configured" in code


def test_injected_eval_wins_over_discovered_conflict():
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    injected = [{"criteria": "correctness", "expected": "perfect", "eval_type": "builtin"}]
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.discover_fact_evals",
               return_value=([_disc_spec("correctness", "correct")], None)):
        gen = TestGenerator(spans, trace_file="t.json", injected_evals=injected)
        gen.analyze()
    correctness = [e for e in gen.evals if e.get("criteria") == "correctness"]
    assert len(correctness) == 1
    assert correctness[0]["expected"] == "perfect"      # injected wins
    assert not correctness[0].get("_discovered")


def test_discovery_disabled_makes_no_call():
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"})]
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.discover_fact_evals") as disc:
        TestGenerator(spans, trace_file="t.json", discover_evals=False).generate_test_code()
    disc.assert_not_called()


# --- CLI --no-discover-evals (Task 6) --------------------------------------------

def test_cli_no_discover_evals_flag_parsed(monkeypatch):
    import sys
    from monocle_test_tools import generate_test

    class _FakeGen:
        _discovery_note = None
        def generate_test_code(self, test_name="test_generated"):
            return "print('ok')"

    # main() does `from ...test_generator import TestGenerator`, binding the real
    # class object; patch the method on that class so the local name is affected.
    with patch("monocle_test_tools.test_generator.TestGenerator.from_json_file",
               return_value=_FakeGen()) as mock_ff:
        monkeypatch.setattr(sys, "argv", ["prog", "trace.json", "--no-discover-evals"])
        rc = generate_test.main()
    assert rc == 0
    assert mock_ff.call_args.kwargs.get("discover_evals") is False


def test_cli_discover_evals_default_on(monkeypatch):
    import sys
    from monocle_test_tools import generate_test

    class _FakeGen:
        _discovery_note = None
        def generate_test_code(self, test_name="test_generated"):
            return "print('ok')"

    with patch("monocle_test_tools.test_generator.TestGenerator.from_json_file",
               return_value=_FakeGen()) as mock_ff:
        monkeypatch.setattr(sys, "argv", ["prog", "trace.json"])
        rc = generate_test.main()
    assert rc == 0
    assert mock_ff.call_args.kwargs.get("discover_evals") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
