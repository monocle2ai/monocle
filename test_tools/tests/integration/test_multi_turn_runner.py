"""Integration test for the multi-turn runner using an in-process fake runner.

The framework-specific integration tests (ADK, Strands, etc.) need a live LLM,
so this test instead swaps in a fake AgentRunner that records the session id it
is handed on every turn and exports a pre-canned set of spans per turn into the
validator's in-memory exporter. That is enough to exercise the real multi-turn
machinery end to end without any network calls:

- one runner instance is reused across all turns (session persistence),
- the same session id is threaded into every turn,
- spans are accumulated across turns (not cleared between them),
- per-turn assertions see only that turn's spans,
- session-level assertions see every turn's spans,
- output of turn n is chained into the input of turn n+1,
- end_session is called once the run completes.
"""
import os

import pytest

from monocle_test_tools import MonocleValidator, MultiTurnTestCase
from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.runner.agent_runner import AgentRunner
import monocle_test_tools.validator as validator_module
from monocle_test_tools.file_span_loader import JSONSpanLoader
from monocle_apptrace.instrumentation.common.utils import get_scopes


def _load_spans():
    here = os.path.dirname(os.path.abspath(__file__))
    trace = os.path.join(here, "..", "unit", "traces", "trace1.json")
    return JSONSpanLoader.from_json(os.path.abspath(trace))


class FakeMultiTurnRunner(AgentRunner):
    """Fake runner that emits canned spans per turn and tracks session lifecycle."""

    def __init__(self, validator, turn_spans, turn_outputs):
        self._validator = validator
        self._turn_spans = turn_spans
        self._turn_outputs = turn_outputs
        self._turn = 0
        self.session_ids_seen = []
        self.inputs_seen = []
        self.end_session_called_with = None

    async def run_agent_async(self, root_agent, *args, session_id: str = None):
        self.session_ids_seen.append(session_id)
        self.inputs_seen.append(args[0] if args else None)
        spans = self._turn_spans[self._turn]
        if spans:
            self._validator.memory_exporter.export(spans)
        output = self._turn_outputs[self._turn]
        self._turn += 1
        return output

    async def end_session(self, session_id: str = None) -> None:
        self.end_session_called_with = session_id


@pytest.mark.asyncio
async def test_multi_turn_runs_all_turns_in_one_session(monkeypatch):
    validator = MonocleValidator()
    validator.cleanup()

    all_spans = _load_spans()
    midpoint = max(1, len(all_spans) // 2)
    turn_spans = [all_spans[:midpoint], all_spans[midpoint:]]
    turn_outputs = ["I can book the flight, which city is the destination?", "Booked to Mumbai"]

    fake = FakeMultiTurnRunner(validator, turn_spans, turn_outputs)
    monkeypatch.setattr(validator_module, "get_agent_runner", lambda t: fake)

    mtc = MultiTurnTestCase(
        session_id="multi_turn_session_test",
        turns=[
            {"test_input": ["Book a flight"]},
            {"test_input": ["The destination is {previous_output}"]},
        ],
    )

    results = await validator.run_multi_turn_agent_async(None, "google_adk", mtc)
    per_turn_spans, outputs, turn_ids = results

    assert fake.session_ids_seen == ["multi_turn_session_test", "multi_turn_session_test"]
    assert "I can book the flight" in fake.inputs_seen[1]
    assert len(per_turn_spans) == 2
    assert turn_ids == ["1", "2"]
    assert len(validator._test_all_up_spans) == len(all_spans)
    assert outputs == turn_outputs
    assert fake.end_session_called_with == "multi_turn_session_test"

    validator.cleanup()


@pytest.mark.asyncio
async def test_multi_turn_auto_assigns_session_id(monkeypatch):
    validator = MonocleValidator()
    validator.cleanup()

    all_spans = _load_spans()
    fake = FakeMultiTurnRunner(validator, [all_spans], ["done"])
    monkeypatch.setattr(validator_module, "get_agent_runner", lambda t: fake)

    mtc = MultiTurnTestCase(turns=[{"test_input": ["hello"]}])
    await validator.test_multi_turn_agent_async(None, "google_adk", mtc)

    assert mtc.session_id is not None
    assert fake.session_ids_seen == [mtc.session_id]

    validator.cleanup()


@pytest.fixture(autouse=True)
def _reset_trace_assertion_state():
    """Keep the class-level assertion list clean around the fluent tests below."""
    TraceAssertion._assertion_errors = []
    yield
    TraceAssertion._assertion_errors = []


class TurnStampingRunner(AgentRunner):
    """Emits one span per call, tagged with the active turn_id scope — the same
    thing the SpanHandler does at runtime — so fluent has_scope can read it back."""

    def __init__(self, validator, spans):
        self._validator = validator
        self._spans = list(spans)
        self._i = 0

    async def run_agent_async(self, root_agent, *args, session_id: str = None):
        span = self._spans[self._i]
        self._i += 1
        turn_id = get_scopes("turn_id").get("turn_id")
        if turn_id is not None:
            span._attributes["scope.turn_id"] = turn_id
        self._validator.memory_exporter.export([span])
        return "ok"


@pytest.mark.asyncio
async def test_auto_turn_id_readable_with_has_scope(monkeypatch):
    """Two run_agent_async calls in one session tag their spans turn "1" and "2",
    read back with the same fluent has_scope agent tests use."""
    validator = MonocleValidator()
    validator.cleanup()
    monkeypatch.setattr(validator_module, "get_agent_runner",
                        lambda t: TurnStampingRunner(validator, _load_spans()))

    await validator.run_agent_async(None, "fake", "turn one", session_id="s")
    await validator.run_agent_async(None, "fake", "turn two", session_id="s")

    asserter = TraceAssertion(filtered_spans=list(validator.spans))
    assert not asserter.has_scope("turn_id", "1").is_assertion_failed
    assert not asserter.has_scope("turn_id", "2").is_assertion_failed
    validator.cleanup()


@pytest.mark.asyncio
async def test_explicit_turn_id_readable_with_has_scope(monkeypatch):
    """An explicit turn_id is used as-is."""
    validator = MonocleValidator()
    validator.cleanup()
    monkeypatch.setattr(validator_module, "get_agent_runner",
                        lambda t: TurnStampingRunner(validator, _load_spans()))

    await validator.run_agent_async(None, "fake", "x", session_id="s", turn_id="book")

    asserter = TraceAssertion(filtered_spans=list(validator.spans))
    assert not asserter.has_scope("turn_id", "book").is_assertion_failed
    validator.cleanup()


@pytest.mark.asyncio
async def test_no_session_has_no_turn_scope(monkeypatch):
    """A run with no session_id is a single turn: its spans carry no turn_id."""
    validator = MonocleValidator()
    validator.cleanup()
    monkeypatch.setattr(validator_module, "get_agent_runner",
                        lambda t: TurnStampingRunner(validator, _load_spans()))

    await validator.run_agent_async(None, "fake", "just one")

    asserter = TraceAssertion(filtered_spans=list(validator.spans))
    assert not asserter.does_not_have_scope("turn_id").is_assertion_failed
    validator.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
