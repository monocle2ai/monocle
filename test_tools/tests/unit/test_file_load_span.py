from asyncio import sleep
import pytest
import os

from monocle_test_tools import TraceAssertion
from test_common.adk_travel_agent import root_agent, root_agent_parallel
TEST_TRACE_FILE = os.path.join(os.path.dirname(__file__), "test_data/monocle_a6567746abaaa45a27be1ab5c7c5aa5f_trace.json")

@pytest.mark.asyncio
async def test_spans_load_from_file(monocle_trace_asserter:TraceAssertion):
    monocle_trace_asserter.with_trace_source("file", trace_path=TEST_TRACE_FILE)
    monocle_trace_asserter.called_agent("okahu_demo_cc_agent_supervisor")
    monocle_trace_asserter.called_tool("okahu_demo_cc_tool_get_product_warranty")
