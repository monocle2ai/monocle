"""Integration test for test generator - demonstrates full workflow."""

import pytest
from pathlib import Path
from monocle_test_tools.test_generator import TestGenerator
from monocle_test_tools import TraceAssertion
from monocle_test_tools.span_loader import JSONSpanLoader


def test_generator_workflow_from_trace1():
    """Test complete workflow: generate test from trace1.json and verify structure."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    # Step 1: Generate test code
    generator = TestGenerator.from_json_file(trace_path)
    test_code = generator.generate_test_code(test_name="test_generated_trace1")
    
    # Verify the generated code structure
    assert "import pytest" in test_code
    assert "from monocle_test_tools import TraceAssertion" in test_code
    assert "def test_generated_trace1" in test_code
    assert "asserter.called_agent" in test_code or "asserter.called_tool" in test_code
    
    # Step 2: Verify we can load and execute assertions (simplified version)
    spans = JSONSpanLoader.from_json(trace_path)
    assert len(spans) > 0, "Should have loaded spans from trace file"


def test_generated_test_trace1(monocle_trace_asserter: TraceAssertion):
    """Example of a generated test running against trace1.json.
    
    This demonstrates what a generated test looks like when executed.
    """
    trace_file = "tests/unit/traces/trace1.json"
    if not Path(trace_file).exists():
        pytest.skip(f"Trace file {trace_file} not found")
    
    # Load the trace
    spans = JSONSpanLoader.from_json(trace_file)
    monocle_trace_asserter.validator.add_remote_spans(spans)
    
    asserter = monocle_trace_asserter
    
    # Agent invocations - adjust based on what's actually in trace1.json
    # This is a generic test that should work for most traces
    asserter.called_agent("adk_hotel_booking_agent_5")


def test_generator_creates_valid_python():
    """Test that generated code is valid Python syntax."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    test_code = generator.generate_test_code()
    
    # Try to compile the generated code
    try:
        compile(test_code, '<generated>', 'exec')
    except SyntaxError as e:
        pytest.fail(f"Generated code has syntax error: {e}")


def test_generator_with_tool_parent():
    """Test that tool assertions include parent agent when available."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    generator.analyze()
    test_code = generator.generate_test_code()
    
    # If tools exist with parent agents, check format
    if generator.tools:
        has_parent = any(agent for agent in generator.tools.values())
        if has_parent:
            # Should have format: called_tool("tool_name", "agent_name")
            assert 'called_tool(' in test_code


def test_multiple_agents_sorted():
    """Test that multiple agents appear in sorted order."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    generator = TestGenerator.from_json_file(trace_path)
    generator.analyze()
    
    if len(generator.agents) > 1:
        test_code = generator.generate_test_code()
        agent_list = sorted(generator.agents)
        
        # Find positions of agents in generated code
        positions = []
        for agent in agent_list:
            if agent in test_code:
                positions.append(test_code.index(agent))
        
        # Positions should be in ascending order (agents appear sorted)
        assert positions == sorted(positions), "Agents should appear in sorted order"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
