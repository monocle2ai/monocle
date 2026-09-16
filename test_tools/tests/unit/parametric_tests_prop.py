from asyncio import sleep
import pytest

from monocle_test_tools import TraceAssertion, get_test_cases
from test_common.adk_travel_agent import root_agent, root_agent_parallel

# Case1: Baseline testing - build golden dataset
# Agent inputs are parametric, the agent behavior validation is in test code
BASELINE_TESTCASES = [
    {
        "input": "Book a flight from San Francisco to Mumbai for 26th Nov 2025",
        "expected" : {
            "outputs" : ["Booked flight", "Mumbai", "San Francisco"], 
        }
    },
    {
        "input": "Book a flight from San Francisco to Los Angeles for 26th Nov 2025",
        "expected" : {
            "output" : ["Booked flight", "Mumbai", "Los Angeles"],
        }
    }
]
@pytest.mark.parametrize("testcase", BASELINE_TESTCASES)
def test_travel_agent_baseline(monocle_trace_asserter:TraceAssertion, testcase):
    # run agent with given input specified in the testcase
    monocle_trace_asserter.run_agent(root_agent, "google_adk", testcase=testcase)

    # assert that the output tokens in the testcase are present in the agent's output
    monocle_trace_asserter.contains_output(testcase=testcase)

    # additional assertions beyond what's in the testcase
    monocle_trace_asserter.called_tool(tool_name="flight_tool", agent_name="fligh_booking_agent")
    monocle_trace_asserter.called_agent(agent_name="supervisor")


# Case 2: Tune eval template =
# You have golden/human curated dataset for a given eval template . You have updated the template and want to ensure that it doesn't regress
EVAL_TUNE_TESTCASES = [
    {
        "input": {"fact_id": "trace1232", "fact_name": "trace"},
        "expected": {
            "evals": {"hallucination": "minor_hallucination"}
        }
    },
    {
        "input": {"fact_id": "trace1232", "fact_name": "trace"},
        "expected": {
            "evals": {"hallucination": "minor_hallucination"}
        }
    }
]
@pytest.mark.parametrize("testcase", EVAL_TUNE_TESTCASES)
def test_eval_tuning(monocle_trace_asserter:TraceAssertion, testcase):
    # Load trace data from fact_id specified in the testcase's input
    monocle_trace_asserter.with_trace_source(source="okahu", testcase=testcase)

    # run eval specified in the test case and compare the result with excepted value from the testcase
    monocle_trace_asserter.check_eval(testcase=testcase)


# Case 3: A/B Testing with local traces
# You are tweaking a knob in the agent application eg fixing a bug by updating agent description,
# You have a baseline traces in locally and want to verify that the change doesn't regress
A_B_LOCAL_TESTCASES = [
    # Monocle API to build the parameterized test cases from the traces in the given local folder
    # This will set the expected output, agents invoked etc in the testcase. That's how we'll be able to 
    # compare agent's original and new behavior for the same input. The traces become the test data.
    get_test_cases(source="file", path="./monocle/test_traces/run_123")
]
@pytest.mark.parametrize("testcase", A_B_LOCAL_TESTCASES)
def test_travel_agent_baseline(monocle_trace_asserter:TraceAssertion, testcase):
    # run agent, the input is extracted from the turn input in the trace, that way you don't need local copy of input data
    monocle_trace_asserter.run_agent(root_agent, "google_adk", testcase=testcase)

    # assert that the output tokens in the testcase are present in the agent's output
    monocle_trace_asserter.contains_output(testcase=testcase)

    # additional assertions beyond what's in the testcase
    monocle_trace_asserter.called_tool(testcase=testcase)
    monocle_trace_asserter.called_agent(testcase=testcase)

# Case 4: A/B Testing with okahu traces
# You are tweaking a knob in the agent application eg introducing a cheaper model,
# You have a baseline traces in locally and want to verify that the change doesn't regress
A_B_OKAHU_TESTCASES = [
    # similar to case 3, but the traces are extracted from Okahu with richer search capabilities, evals and different facts.
    # Okahu becores the persisten test data management
    get_test_cases(source="okahu", start_time="2026-05-29T01:56:37.695019Z",
        end_time = "2026-05-29T01:56:42.704952Z", workflow = "wf", filter_fact="test_runid=t_123")

]
@pytest.mark.parametrize("testcase", BASELINE_TESTCASES)
def test_travel_agent_baseline(monocle_trace_asserter:TraceAssertion, testcase):
    # Same test as Case #3 now runing with Okahu stored traces.
    pass