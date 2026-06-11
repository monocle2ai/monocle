import pytest
from monocle_test_tools import TraceAssertion
from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

# Update these constants with real values from your Okahu environment to run the tests
DEMO_WORKFLOW_NAME = "Okahu-Loader-Demo"
PLACEHOLDER_TRACE_ID = "642dbd9d0dfcfdbdc8849f67f34c8a19"

@pytest.fixture()
def monocle_trace_asserter():
    asserter = TraceAssertion()
    asserter.cleanup()
    asserter.validator._trace_source = "okahu"
    yield asserter
    asserter.cleanup()


def test_agent_tool_and_io_assertions(monocle_trace_asserter: TraceAssertion):
    monocle_trace_asserter.load_spans(OkahuSpanLoader.get_spans(DEMO_WORKFLOW_NAME, PLACEHOLDER_TRACE_ID))

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

def test_performance_and_eval(monocle_trace_asserter: TraceAssertion):
    monocle_trace_asserter.load_spans(OkahuSpanLoader.get_spans(DEMO_WORKFLOW_NAME, PLACEHOLDER_TRACE_ID))

    # Performance: token usage and workflow duration
    monocle_trace_asserter.under_token_limit(10000)
    monocle_trace_asserter.under_duration(30.0, "seconds")

    # Hallucination eval
    monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")
