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


def _make_mock_span(span_type, entity_name="", entity_type="", events=None, attrs=None):
    """Build a minimal mock span object for generator tests."""
    from unittest.mock import MagicMock
    span = MagicMock()
    base_attrs = {"span.type": span_type}
    if entity_name:
        base_attrs["entity.1.name"] = entity_name
    if entity_type:
        base_attrs["entity.1.type"] = entity_type
    if attrs:
        base_attrs.update(attrs)
    span.attributes = base_attrs
    span.events = events or []
    span.start_time = None
    span.end_time = None
    return span


def _make_event(name, **attributes):
    from unittest.mock import MagicMock
    ev = MagicMock()
    ev.name = name
    ev.attributes = attributes
    return ev


# --- Issue #714: has_attribute assertions in generated code ---

def test_generate_emits_has_attribute_for_entity_type():
    """Generator emits has_attribute() for notable span attributes (issue #714)."""
    span = _make_mock_span(
        "agentic.invocation",
        entity_name="my_agent",
        entity_type="agent.langgraph",
    )
    gen = TestGenerator(spans=[span])
    code = gen.generate_test_code()

    assert "has_attribute" in code
    assert "entity.1.type" in code
    assert "agent.langgraph" in code


def test_analyze_populates_span_attributes():
    """analyze() collects notable attributes into span_attributes dict (issue #714)."""
    span = _make_mock_span(
        "agentic.tool.invocation",
        entity_name="my_tool",
        entity_type="tool.openai",
    )
    gen = TestGenerator(spans=[span])
    gen.analyze()

    assert "agentic.tool.invocation" in gen.span_attributes
    attrs = gen.span_attributes["agentic.tool.invocation"]
    assert attrs.get("entity.1.type") == "tool.openai"


def test_span_attributes_reset_on_repeated_analyze():
    """Repeated analyze() calls do not accumulate duplicate attribute values (issue #714)."""
    span = _make_mock_span("agentic.invocation", entity_name="a", entity_type="agent.x")
    gen = TestGenerator(spans=[span])
    gen.analyze()
    first = dict(gen.span_attributes)
    gen.analyze()
    assert gen.span_attributes == first


# --- Issue #687: additional assertion types in generated code ---

def test_generate_emits_has_input_for_agent():
    """Generator emits .has_input() chain for agent with captured input (issue #687)."""
    input_ev = _make_event("data.input", input="What is the weather today?")
    span = _make_mock_span("agentic.invocation", entity_name="weather_agent", events=[input_ev])
    gen = TestGenerator(spans=[span])
    code = gen.generate_test_code()

    assert "has_input" in code
    assert "What is the weather today?" in code


def test_generate_emits_tool_input_output():
    """Generator emits .has_input()/.has_output() for tools with event data (issue #687)."""
    inp_ev = _make_event("data.input", input="search query")
    out_ev = _make_event("data.output", response="search result")
    span = _make_mock_span(
        "agentic.tool.invocation",
        entity_name="search_tool",
        events=[inp_ev, out_ev],
    )
    gen = TestGenerator(spans=[span])
    code = gen.generate_test_code()

    assert "search query" in code
    assert "search result" in code
    assert "has_input" in code
    assert "has_output" in code


def test_analyze_populates_tool_inputs_outputs():
    """analyze() captures first tool input/output snippets (issue #687)."""
    inp_ev = _make_event("data.input", input="my tool input")
    out_ev = _make_event("data.output", response="my tool output")
    span = _make_mock_span("agentic.tool.invocation", entity_name="t1", events=[inp_ev, out_ev])
    gen = TestGenerator(spans=[span])
    gen.analyze()

    assert gen.tool_inputs.get("t1") == "my tool input"
    assert gen.tool_outputs.get("t1") == "my tool output"


def test_analyze_populates_agent_inputs():
    """analyze() captures agent input snippets (issue #687)."""
    inp_ev = _make_event("data.input", input="Hello agent!")
    span = _make_mock_span("agentic.invocation", entity_name="bot", events=[inp_ev])
    gen = TestGenerator(spans=[span])
    gen.analyze()

    assert gen.agent_inputs.get("bot") == ["Hello agent!"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
