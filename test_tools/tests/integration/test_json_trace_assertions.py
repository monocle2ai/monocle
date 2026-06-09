"""Load spans from a local JSON trace file and run assertions.
See docs/monocle_trace_loading_guide.md for setup and usage.
"""
import json
import pytest
import os
from monocle_test_tools import TraceAssertion
from monocle_test_tools.file_span_loader import JSONSpanLoader

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def load_test_spans(relative_path: str) -> list:
    full_path = os.path.join(_TEST_DIR, relative_path)
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [JSONSpanLoader._from_dict(s) for s in data.get("spans", [])]


@pytest.fixture()
def monocle_trace_asserter():
    asserter = TraceAssertion()
    asserter.cleanup()
    asserter.validator._trace_source = "local"
    yield asserter
    asserter.cleanup()


def test_agent_tool_and_io_assertions(monocle_trace_asserter: TraceAssertion):
    """Agent/tool invocation, input/output, and hallucination eval."""
    monocle_trace_asserter.load_spans(load_test_spans("json_loader_traces/642dbd9d0dfcfdbdc8849f67f34c8a19.json"))


    # Verify expected agents were invoked (and one was not)
    monocle_trace_asserter.called_agent("okahu_demo_cc_agent_supervisor")
    monocle_trace_asserter.called_agent("okahu_demo_cc_agent_refund")
    monocle_trace_asserter.does_not_call_agent("okahu_demo_lg_agent_lodging_assistant")

    # Verify tool calls, including which agent called which tool
    monocle_trace_asserter.called_tool("okahu_demo_cc_tool_lookup_order", "okahu_demo_cc_agent_order_lookup")
    monocle_trace_asserter.called_tool("okahu_demo_cc_tool_process_refund")
    monocle_trace_asserter.does_not_call_tool("okahu_demo_cc_tool_get_return_policy")

    # Check input/output contain (or don't contain) expected substrings
    monocle_trace_asserter.contains_input("ORD-STD-0350")
    monocle_trace_asserter.does_not_contain_input("ORD-A1042")
    monocle_trace_asserter.contains_output("350")
    monocle_trace_asserter.does_not_contain_output("ORD-B1042")

    # Hallucination eval
    monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")


def test_performance_and_eval(monocle_trace_asserter: TraceAssertion):
    """Token limit, duration, tool-level output, and hallucination eval."""
    monocle_trace_asserter.load_spans(load_test_spans("json_loader_traces/be1db02405333e449c16248d4b7a5057.json"))


    # Performance: token usage and workflow duration
    monocle_trace_asserter.under_token_limit(10000)
    monocle_trace_asserter.under_duration(30.0, "seconds")

    # Hallucination eval
    monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")
