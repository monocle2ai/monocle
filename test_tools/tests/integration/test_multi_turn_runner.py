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
from monocle_test_tools.runner.agent_runner import AgentRunner
import monocle_test_tools.validator as validator_module
from monocle_test_tools.file_span_loader import JSONSpanLoader


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
    per_turn_spans, outputs = results

    assert fake.session_ids_seen == ["multi_turn_session_test", "multi_turn_session_test"]
    assert "I can book the flight" in fake.inputs_seen[1]
    assert len(per_turn_spans) == 2
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


if __name__ == "__main__":
    pytest.main([__file__])
