"""Unit tests for TestGenerator functionality."""

import pytest
import tempfile
from pathlib import Path
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
    # No "Option" scaffolding when a single source is requested.
    assert "Option 2" not in test_code


def test_trace_source_okahu_only():
    """When trace_source='okahu', only the okahu loader is generated (as active code)."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")

    generator = TestGenerator.from_json_file(trace_path, trace_source="okahu")
    test_code = generator.generate_test_code()

    assert 'monocle_trace_asserter.with_trace_source("okahu"' in test_code
    assert 'with_trace_source("file"' not in test_code


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
