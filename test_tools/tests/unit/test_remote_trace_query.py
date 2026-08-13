"""Validator-level tests for fact-based remote trace retrieval.

A runner whose spans are produced in another process names the fact that
identifies them via ``get_remote_trace_query()``; the validator passes that to
``import_traces`` so the spans join the ones assertions run against. Okahu is
stubbed, so these run without network access or credentials.
"""
import os

import pytest

import monocle_test_tools.validator as validator_module
from monocle_test_tools import MonocleValidator
from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.runner.agent_runner import AgentRunner
from monocle_test_tools.file_span_loader import JSONSpanLoader

SESSION_ID = "monocle_test_session_" + "c" * 32
REMOTE_WORKFLOW = "deployed_agent_workflow"


def _load_spans():
    here = os.path.dirname(os.path.abspath(__file__))
    return JSONSpanLoader.from_json(os.path.join(here, "traces", "trace1.json"))


class SessionCorrelatedRunner(AgentRunner):
    """Stand-in for a runner whose spans are produced remotely."""

    def __init__(self, session_id=SESSION_ID, workflow=REMOTE_WORKFLOW):
        self._session_id = session_id
        self._workflow = workflow

    def run_agent(self, root_agent, *args, **kwargs):
        return "remote response"

    async def run_agent_async(self, root_agent, *args, session_id=None, **kwargs):
        return "remote response"

    def get_remote_traces_source(self):
        return "okahu"

    def get_remote_trace_query(self):
        return {"id": self._session_id, "fact_name": "session",
                "workflow_name": self._workflow}


@pytest.fixture
def validator():
    v = MonocleValidator()
    v.cleanup()
    yield v
    v.cleanup()


def test_query_is_forwarded_to_import_traces(validator, monkeypatch):
    """The runner's query reaches import_traces verbatim."""
    runner = SessionCorrelatedRunner()
    monkeypatch.setattr(validator_module, "get_agent_runner", lambda t: runner)

    captured = {}
    monkeypatch.setattr(MonocleValidator, "import_traces",
                        lambda self, source, **kw: captured.update(source=source, **kw))

    validator.run_agent(None, "any_type", "hello")

    assert captured["source"] == "okahu"
    assert captured["id"] == SESSION_ID
    assert captured["fact_name"] == "session"
    assert captured["workflow_name"] == REMOTE_WORKFLOW


def test_remote_spans_reach_assertions(validator, monkeypatch):
    """Spans imported by the query are visible to the fluent assertions."""
    spans = _load_spans()
    runner = SessionCorrelatedRunner()
    monkeypatch.setattr(validator_module, "get_agent_runner", lambda t: runner)
    # Stand in for Okahu returning the deployed agent's spans for this session.
    monkeypatch.setattr(MonocleValidator, "import_traces",
                        lambda self, source, **kw: self.add_remote_spans(spans))

    validator.run_agent(None, "any_type", "hello")

    assert len(validator.spans) == len(spans)
    asserter = TraceAssertion()
    tool_spans = validator._get_all_tool_invocation_spans(filtered_spans=validator.spans)
    assert tool_spans, "remote tool spans should be assertable"


def test_runners_without_a_query_keep_trace_id_lookup(validator, monkeypatch):
    """Default runners are unaffected: no query, so import_traces gets none."""

    class PlainRunner(AgentRunner):
        def run_agent(self, root_agent, *args, **kwargs):
            return "local response"

        def get_remote_traces_source(self):
            return "okahu"

    monkeypatch.setattr(validator_module, "get_agent_runner", lambda t: PlainRunner())
    captured = {}
    monkeypatch.setattr(MonocleValidator, "import_traces",
                        lambda self, source, **kw: captured.update(source=source, kwargs=kw))

    validator.run_agent(None, "any_type", "hello")

    assert captured["source"] == "okahu"
    assert captured["kwargs"] == {}, "trace-id lookup must receive no extra arguments"


def test_base_runner_query_is_empty():
    assert AgentRunner().get_remote_trace_query() == {}


def test_not_found_is_retried_then_surfaced(validator, monkeypatch):
    """A fact lookup keeps polling while spans are still in flight.

    The span loader raises ConnectionError until the remote spans land, so the
    fact path must retry rather than fail on the first attempt.
    """
    monkeypatch.setattr(validator_module, "REMOTE_FACT_TIMEOUT_SECONDS", 3)
    attempts = {"n": 0}

    def flaky(self, source, **kw):
        attempts["n"] += 1
        raise ConnectionError("No traces found")

    monkeypatch.setattr(MonocleValidator, "import_traces", flaky)
    validator._trace_source = "okahu"

    with pytest.raises(ConnectionError):
        validator._fetch_remote_traces(id=SESSION_ID, fact_name="session",
                                       workflow_name=REMOTE_WORKFLOW)

    assert attempts["n"] > 1, "should have retried while spans were in flight"


def test_trace_id_path_does_not_retry_connection_errors(validator, monkeypatch):
    """Existing behaviour preserved: the trace-id path is unchanged."""
    attempts = {"n": 0}

    def boom(self, source, **kw):
        attempts["n"] += 1
        raise ConnectionError("backend down")

    monkeypatch.setattr(MonocleValidator, "import_traces", boom)
    validator._trace_source = "okahu"

    with pytest.raises(ConnectionError):
        validator._fetch_remote_traces()

    assert attempts["n"] == 1, "trace-id path must not start retrying ConnectionError"


if __name__ == "__main__":
    pytest.main([__file__])
