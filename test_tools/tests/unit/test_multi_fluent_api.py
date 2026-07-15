import os
import pytest 
from monocle_test_tools import TraceAssertion
from span_loader import JSONSpanLoader
os.environ["MONOCLE_EXPORT_FAILED_TESTS_ONLY"] = "true"

def test_singleton_checks(monocle_trace_asserter:TraceAssertion):
    """Test that the correct tool is invoked and returns expected output."""
    monocle_trace_asserter.load_spans(JSONSpanLoader.load_spans("traces/trace3.json"))

    monocle_trace_asserter.called_tool("get_nba_past_scores")
    monocle_trace_asserter.called_agent("Nba Score Agent")

def test_negative_singleton_checks(monocle_trace_asserter:TraceAssertion):
    """Test that incorrect tool/agent invocations are caught."""
    monocle_trace_asserter.load_spans(JSONSpanLoader.load_spans("traces/trace3.json"))

    monocle_trace_asserter.does_not_call_tool("non_existent_tool")
    monocle_trace_asserter.does_not_call_agent("Non Existent Agent")

def test_input_output_checks(monocle_trace_asserter:TraceAssertion):
    monocle_trace_asserter.load_spans(JSONSpanLoader.load_spans("traces/trace3.json"))

    monocle_trace_asserter.called_tool("get_nba_past_scores").contains_any_input("Clippers","Hornets").contains_output("Hornets")
    monocle_trace_asserter.called_agent("Nba Score Agent").has_any_input("What happened in Clippers game on 22 Nov 2025", "foo bar").contains_any_output("131-116","baz qux")

def test_negative_input_output_checks(monocle_trace_asserter:TraceAssertion):
    monocle_trace_asserter.load_spans(JSONSpanLoader.load_spans("traces/trace3.json"))

    monocle_trace_asserter.called_tool("get_nba_past_scores").does_not_contain_any_input("Lakers","Bulls").does_not_contain_any_output("Lakers", "Bulls")
    monocle_trace_asserter.called_agent("Nba Score Agent").does_not_have_any_input("This input does not exist", "foo bar").does_not_contain_any_output("This output does not exist", "baz qux")

if __name__ == "__main__":
    pytest.main([__file__])