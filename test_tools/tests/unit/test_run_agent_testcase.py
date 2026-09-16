"""run_agent/run_agent_async driven by a FluentTestCase.

The interesting case is a FactID input: the recorded trace is fetched only to
recover the prompt it was run with, that prompt is replayed against a live
agent, and the fetched spans are thrown away.
"""
import os

import pytest

from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.span_loader import JSONSpanLoader

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
TRACE3_TURN_INPUT = "What happened in Clippers game on 22 Nov 2025"


@pytest.fixture(autouse=True)
def _reset_trace_assertion_class_state():
    TraceAssertion._assertion_errors = []
    yield
    TraceAssertion._assertion_errors = []


@pytest.fixture(name="asserter")
def asserter_fixture(monkeypatch):
    """An asserter whose validator records run_agent calls instead of running one."""
    asserter = TraceAssertion()
    calls = []

    def fake_run_agent(agent, agent_type, *args, **kwargs):
        calls.append({"agent": agent, "agent_type": agent_type,
                      "args": args, "kwargs": kwargs})
        return "ran"

    async def fake_run_agent_async(agent, agent_type, *args, **kwargs):
        return fake_run_agent(agent, agent_type, *args, **kwargs)

    monkeypatch.setattr(asserter.validator, "run_agent", fake_run_agent)
    monkeypatch.setattr(asserter.validator, "run_agent_async", fake_run_agent_async)
    asserter.calls = calls
    return asserter


@pytest.fixture(name="fetched")
def fetched_fixture(monkeypatch):
    """Make import_traces a pure fetch of trace3.json, recording its kwargs."""
    spans = JSONSpanLoader.from_json(os.path.join(TRACES_DIR, "trace3.json"))
    seen = {}

    def fake_import_traces(self, **kwargs):
        seen.update(kwargs)
        return spans

    monkeypatch.setattr("monocle_test_tools.validator.MonocleValidator.import_traces",
                        fake_import_traces)
    return seen


def test_tuple_input_is_splatted(asserter):
    asserter.run_agent("agent", "langgraph", testcase={"input": "Book a flight"})

    assert asserter.calls[0]["args"] == ("Book a flight",)


def test_result_is_returned(asserter):
    result = asserter.run_agent("agent", "langgraph", testcase={"input": "go"})

    assert result == "ran"


def test_factid_input_replays_the_recorded_prompt(asserter, fetched):
    asserter.run_agent("agent", "langgraph",
                       testcase={"input": {"fact_id": "cc777", "source": "file"}})

    assert asserter.calls[0]["args"] == (TRACE3_TURN_INPUT,)


def test_factid_fetch_does_not_load_spans(asserter, fetched):
    asserter.run_agent("agent", "langgraph",
                       testcase={"input": {"fact_id": "cc777", "source": "file"}})

    assert fetched["load_spans"] is False


def test_factid_fetch_uses_the_mapped_kwargs(asserter, fetched):
    asserter.run_agent("agent", "langgraph",
                       testcase={"input": {"fact_id": "cc777", "source": "okahu",
                                           "fact_name": "session"}})

    assert fetched["trace_source"] == "okahu"
    assert fetched["id"] == "cc777"
    assert fetched["fact_name"] == "session"


def test_positional_args_with_testcase_raises(asserter):
    with pytest.raises(ValueError, match="cannot be combined with 'testcase'"):
        asserter.run_agent("agent", "langgraph", "extra", testcase={"input": "go"})


def test_missing_input_raises(asserter):
    with pytest.raises(ValueError, match="no input to run"):
        asserter.run_agent("agent", "langgraph", testcase={"evals": {"a": "x"}})


def test_without_testcase_behaves_as_before(asserter):
    asserter.run_agent("agent", "langgraph", "plain input")

    assert asserter.calls[0]["args"] == ("plain input",)


@pytest.mark.asyncio
async def test_async_tuple_input_is_splatted(asserter):
    await asserter.run_agent_async("agent", "google_adk", testcase={"input": "go"})

    assert asserter.calls[0]["args"] == ("go",)


@pytest.mark.asyncio
async def test_async_factid_input_replays_the_recorded_prompt(asserter, fetched):
    await asserter.run_agent_async(
        "agent", "google_adk",
        testcase={"input": {"fact_id": "cc777", "source": "file"}})

    assert asserter.calls[0]["args"] == (TRACE3_TURN_INPUT,)


@pytest.mark.asyncio
async def test_async_positional_args_with_testcase_raises(asserter):
    with pytest.raises(ValueError, match="cannot be combined with 'testcase'"):
        await asserter.run_agent_async("agent", "google_adk", "extra",
                                       testcase={"input": "go"})
