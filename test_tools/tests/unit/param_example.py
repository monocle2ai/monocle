from asyncio import sleep
import pytest

from monocle_test_tools import TraceAssertion, get_test_cases
from test_common.adk_travel_agent import root_agent, root_agent_parallel

FLIGHT_AGENT_TESTS =[
    {
        "name": "test1",
        "input": "Book a flight from San Francisco to Mumbai for 26th Nov 2025",
        "expected" : {
            "output" : "Booked flight",
            "agents": {
                "supervisor" : {"output": "booked Flight"},
                "flight_agent": {"input": "Book a flight from San Francisco to Mumbai for 26th Nov 2025"}
            },
            "tools": {
                "flight_tool": {}
            },
            "evals": {
                "hallucination": "minor_hallucination"
            }
        }
    },
]
@pytest.mark.parametrize("testcase", FLIGHT_AGENT_TESTS)
@pytest.mark.asyncio
async def test_tool_invocation(monocle_trace_asserter:TraceAssertion, testcase):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", 
                        testcase["input"])

    monocle_trace_asserter.called_tool(testcase=testcase)
    monocle_trace_asserter.called_agent(testcase=testcase).contains_output(testcase=testcase)
    monocle_trace_asserter.under_token_limit(10000)


EVAL_TUNING_TEST =[
    {
        "input": {"fact_id": "trace1232", "fact_name": "trace"},
        "expected": {
            "evals": {"hallucination": "minor_hallucination", "fustration": "not_frustrated"}
        }
    },
    {
        "input": {"fact_id": "trace1232", "fact_name": "trace"},
        "evals": {"hallucination": "major_hallucination", "fustration": "not_frustrated"}
    },
]
@pytest.mark.parametrize("testcase", FLIGHT_AGENT_TESTS)
@pytest.mark.asyncio
async def test_tool_invocation(monocle_trace_asserter:TraceAssertion, testcase):
    """ Compute new evals from existing data to generate golden dataset of evals for template tunning"""
    await monocle_trace_asserter.with_trace_source(source="okahu", testcase=testcase, workflow_name="wf")
    monocle_trace_asserter.check_eval(testcase=testcase)


EVAL_REGRESSION_TEST= [
    {
        "input": {"fact_id": "trace1232", "fact_name": "trace"},
    },
    {
        "input": {"fact_id": "trace1232", "fact_name": "trace"},
    },
]
@pytest.mark.parametrize("testcase", FLIGHT_AGENT_TESTS)
@pytest.mark.asyncio
async def test_tool_invocation(monocle_trace_asserter:TraceAssertion, testcase):
    """ 
    Verify the existing evals on the golden data not broken by updates to eval template.
    Note that the expected values are populated by Monocle from the trace being loaded.
    """
    await monocle_trace_asserter.with_trace_source(source="okahu", testcase=testcase, workflow_name="wf")
    monocle_trace_asserter.check_eval(testcase=testcase)

AGENT_AB_TESTING = get_test_cases(source="okahu", start_time="2026-05-29T01:56:37.695019Z",
    end_time = "2026-05-29T01:56:42.704952Z", workflow = "wf", filter_fact="test_runid=t_123")

@pytest.mark.parametrize("testcase", FLIGHT_AGENT_TESTS)
@pytest.mark.asyncio
async def test_tool_invocation(monocle_trace_asserter:TraceAssertion, testcase):
    """ 
    Verify the existing evals on the golden data not broken by updates to eval template.
    The expected values are populated by Monocle from the trace being loaded.
    The input from that trace/turn is extracted and pass to the run_agent() so it can be a A/B test
    """
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", 
                        testcase=testcase)

    monocle_trace_asserter.called_tool(testcase=testcase)
    monocle_trace_asserter.called_agent(testcase=testcase).contains_output(testcase=testcase)
    monocle_trace_asserter.under_token_limit(10000)