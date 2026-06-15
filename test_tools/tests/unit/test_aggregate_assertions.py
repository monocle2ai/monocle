"""Unit tests for aggregate assertion functionality."""

import pytest
from pathlib import Path
from monocle_test_tools import TraceAssertion
from monocle_test_tools.span_loader import JSONSpanLoader


@pytest.fixture
def trace_asserter_with_data(monocle_trace_asserter: TraceAssertion):
    """Load test trace with multiple agent and tool invocations."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    monocle_trace_asserter.load_spans(JSONSpanLoader.from_json(trace_path))
    return monocle_trace_asserter


def test_called_agent_with_exact_count(trace_asserter_with_data):
    """Test called_agent with exact count parameter."""
    # This should pass if the agent was called exactly once
    trace_asserter_with_data.called_agent("adk_hotel_booking_agent_5", count=1)


def test_called_agent_with_min_count(trace_asserter_with_data):
    """Test called_agent with min_count parameter."""
    # This should pass if agent was called at least once
    trace_asserter_with_data.called_agent("adk_hotel_booking_agent_5", min_count=1)


def test_called_agent_with_max_count(trace_asserter_with_data):
    """Test called_agent with max_count parameter."""
    # This should pass if agent was called at most 5 times
    trace_asserter_with_data.called_agent("adk_hotel_booking_agent_5", max_count=5)


def test_called_agent_with_range(trace_asserter_with_data):
    """Test called_agent with min and max count parameters."""
    # This should pass if agent was called between 1-10 times
    trace_asserter_with_data.called_agent("adk_hotel_booking_agent_5", min_count=1, max_count=10)


def test_called_agent_invalid_params_raises_error():
    """Test that specifying both count and min/max raises ValueError."""
    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")
    
    from monocle_test_tools.fluent_api import TraceAssertion as TA
    asserter = TA()
    asserter.load_spans(JSONSpanLoader.from_json(trace_path))
    
    with pytest.raises(ValueError, match="Cannot specify both"):
        asserter.called_agent("adk_hotel_booking_agent_5", count=1, min_count=1)


def test_called_tool_with_exact_count(trace_asserter_with_data):
    """Test called_tool with exact count parameter."""
    # Get first tool from trace
    tool_spans = trace_asserter_with_data.validator._get_all_tool_invocation_spans()
    if tool_spans:
        tool_name = tool_spans[0].attributes.get("entity.1.name")
        trace_asserter_with_data.called_tool(tool_name, count=1)


def test_called_tool_with_min_count(trace_asserter_with_data):
    """Test called_tool with min_count parameter."""
    tool_spans = trace_asserter_with_data.validator._get_all_tool_invocation_spans()
    if tool_spans:
        tool_name = tool_spans[0].attributes.get("entity.1.name")
        trace_asserter_with_data.called_tool(tool_name, min_count=1)


def test_called_tool_with_agent_and_count(trace_asserter_with_data):
    """Test called_tool with both agent_name and count parameters."""
    tool_spans = trace_asserter_with_data.validator._get_all_tool_invocation_spans()
    if tool_spans:
        tool_name = tool_spans[0].attributes.get("entity.1.name")
        agent_name = tool_spans[0].attributes.get("entity.2.name")
        if agent_name:
            trace_asserter_with_data.called_tool(tool_name, agent_name=agent_name, count=1)


def test_called_agents_with_exact_count(trace_asserter_with_data):
    """Test called_agents with exact count of all agent invocations."""
    # Get actual count
    agent_spans = trace_asserter_with_data.validator._get_all_agent_invocation_spans()
    actual_count = len(agent_spans)
    
    if actual_count > 0:
        trace_asserter_with_data.called_agents(count=actual_count)


def test_called_agents_with_min_count(trace_asserter_with_data):
    """Test called_agents with min_count parameter."""
    # Should pass if there's at least 1 agent invocation
    trace_asserter_with_data.called_agents(min_count=1)


def test_called_agents_with_max_count(trace_asserter_with_data):
    """Test called_agents with max_count parameter."""
    # Should pass with a high enough max_count
    trace_asserter_with_data.called_agents(max_count=100)


def test_called_agents_with_range(trace_asserter_with_data):
    """Test called_agents with both min and max count."""
    trace_asserter_with_data.called_agents(min_count=1, max_count=50)


def test_called_tools_with_exact_count(trace_asserter_with_data):
    """Test called_tools with exact count of all tool invocations."""
    # Get actual count
    tool_spans = trace_asserter_with_data.validator._get_all_tool_invocation_spans()
    actual_count = len(tool_spans)
    
    if actual_count > 0:
        trace_asserter_with_data.called_tools(count=actual_count)


def test_called_tools_with_min_count(trace_asserter_with_data):
    """Test called_tools with min_count parameter."""
    tool_spans = trace_asserter_with_data.validator._get_all_tool_invocation_spans()
    if len(tool_spans) > 0:
        trace_asserter_with_data.called_tools(min_count=1)


def test_called_tools_with_max_count(trace_asserter_with_data):
    """Test called_tools with max_count parameter."""
    # Should pass with a high enough max_count
    trace_asserter_with_data.called_tools(max_count=100)


def test_called_tools_with_range(trace_asserter_with_data):
    """Test called_tools with both min and max count."""
    tool_spans = trace_asserter_with_data.validator._get_all_tool_invocation_spans()
    if len(tool_spans) > 0:
        trace_asserter_with_data.called_tools(min_count=1, max_count=50)


def test_aggregate_assertions_work_together(trace_asserter_with_data):
    """Test that multiple aggregate assertions can be chained."""
    agent_spans = trace_asserter_with_data.validator._get_all_agent_invocation_spans()
    tool_spans = trace_asserter_with_data.validator._get_all_tool_invocation_spans()
    
    if len(agent_spans) > 0 and len(tool_spans) > 0:
        trace_asserter_with_data \
            .called_agents(min_count=1) \
            .called_tools(min_count=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
