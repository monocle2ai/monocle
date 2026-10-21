"""The two flows stage 1 exists to make writable, end to end.

These are the spec's success criteria: a parametrize dict goes in unconverted
and drives the whole test. Okahu and the agent runner are stubbed -- what is
under test is the wiring, not the network.
"""
import os
from unittest.mock import MagicMock

import pytest

from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.span_loader import JSONSpanLoader

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
TRACE3_TURN_INPUT = "What happened in Clippers game on 22 Nov 2025"

EVAL_TUNE_TESTCASES = [
    {
        "input": {"fact_id": "trace1232", "fact_name": "trace"},
        "expected": {"evals": {"hallucination": "minor_hallucination"}},
    },
    {
        "input": {"fact_id": "trace1232", "fact_name": "trace"},
        "expected": {"evals": {"hallucination": "minor_hallucination"}},
    },
]


@pytest.fixture(autouse=True)
def _reset_trace_assertion_class_state():
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._eval_stashes = []
    TraceAssertion._okahu_filter = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._eval_stashes = []
    TraceAssertion._okahu_filter = None


@pytest.fixture(name="spans")
def spans_fixture():
    return JSONSpanLoader.from_json(os.path.join(TRACES_DIR, "trace3.json"))


@pytest.fixture(name="asserter")
def asserter_fixture(monkeypatch, spans):
    """An asserter with a stubbed okahu import and a stubbed evaluator."""
    asserter = TraceAssertion(filtered_spans=spans)

    def fake_import_traces(self, **kwargs):
        return spans

    monkeypatch.setattr("monocle_test_tools.validator.MonocleValidator.import_traces",
                        fake_import_traces)

    eval_mock = MagicMock()
    eval_mock.last_fact_results = None
    eval_mock.evaluate.return_value = ("minor_hallucination", "because")
    asserter._eval = eval_mock
    return asserter


@pytest.mark.parametrize("testcase", EVAL_TUNE_TESTCASES)
def test_eval_tuning(asserter, testcase):
    """The spec's first success criterion, verbatim."""
    asserter.with_trace_source(source="okahu", testcase=testcase)
    asserter.check_eval(testcase=testcase)

    assert not TraceAssertion._assertion_errors


@pytest.mark.parametrize("testcase", EVAL_TUNE_TESTCASES)
@pytest.mark.asyncio
async def test_ab_replay(asserter, monkeypatch, testcase):
    """The spec's second success criterion: replay the recorded input, re-eval."""
    ran_with = []

    async def fake_run_agent_async(agent, agent_type, *args, **kwargs):
        ran_with.append(args)
        return "ran"

    monkeypatch.setattr(asserter.validator, "run_agent_async", fake_run_agent_async)

    await asserter.run_agent_async("root_agent", "google_adk", testcase=testcase)
    asserter.check_eval(testcase=testcase)

    assert ran_with == [(TRACE3_TURN_INPUT,)]
    assert not TraceAssertion._assertion_errors


def test_a_failing_eval_is_reported(asserter):
    asserter._eval.evaluate.return_value = ("major_hallucination", "drifted")

    result = asserter.check_eval(testcase=EVAL_TUNE_TESTCASES[0])

    assert result.has_assertions()
    assert "major_hallucination" in result.get_assertion_messages()
    TraceAssertion._assertion_errors = []
