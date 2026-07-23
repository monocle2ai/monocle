"""Unit tests for TestGenerator functionality."""

import pytest
import tempfile
from pathlib import Path
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


from types import SimpleNamespace


def _event(name, attributes):
    return SimpleNamespace(name=name, attributes=attributes)


def _span(attributes, events=(), trace_id=0xABC123):
    return SimpleNamespace(
        attributes=attributes,
        events=list(events),
        start_time=None,
        end_time=None,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
