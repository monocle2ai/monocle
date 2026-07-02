"""Regression tests for the OpenAI Agents SDK (0.17.x) instrumentation fixes:
inference auto-close sentinel, run_single_turn wrapping/agent resolution, turn-collapse."""
from types import SimpleNamespace

from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.common.scope_wrapper import start_scope, stop_scope
from monocle_apptrace.instrumentation.metamodel.openai.entities.inference import INFERENCE
from monocle_apptrace.instrumentation.metamodel.agents import _helper, methods
from monocle_apptrace.instrumentation.metamodel.agents.agents_processor import (
    AgentsSpanHandler,
    AGENT_TURN_SCOPE,
)
from monocle_apptrace.instrumentation.metamodel.agents.entities.inference import (
    AGENT,
    AGENT_REQUEST,
)


# is_auto_close must close for the Omit()/NotGiven stream sentinel (falsy but not False).
def test_inference_is_auto_close_handles_stream_sentinels():
    is_auto_close = INFERENCE["is_auto_close"]

    from openai import Omit
    from openai._types import NOT_GIVEN

    assert is_auto_close({}) is True
    assert is_auto_close({"stream": False}) is True
    assert is_auto_close({"stream": None}) is True
    assert is_auto_close({"stream": Omit()}) is True
    assert is_auto_close({"stream": NOT_GIVEN}) is True
    # genuine streaming -> defer closing to the stream processor
    assert is_auto_close({"stream": True}) is False


# Agent resolution must handle bindings.execution_agent plus the legacy positional/kwargs shapes.
def test_resolve_agent_instance_from_bindings():
    agent = SimpleNamespace(name="WebSearchAgent", instructions="search", description="d")
    bindings = SimpleNamespace(execution_agent=agent)

    assert _helper.resolve_agent_instance((), {"bindings": bindings}) is agent
    assert _helper.get_agent_name((), {"bindings": bindings}) == "WebSearchAgent"
    assert _helper.get_agent_instructions({"args": (), "kwargs": {"bindings": bindings}}) == "search"

    # legacy positional (Runner.run(agent, input)) and kwargs['agent'] shapes still work
    assert _helper.resolve_agent_instance((agent, "hi"), {}) is agent
    assert _helper.resolve_agent_instance((), {"agent": agent}) is agent

    assert _helper.resolve_agent_instance((), {}) is None
    assert _helper.get_agent_name((), {}) == "Unknown Agent"


# The invocation entry wraps module-level run_single_turn (object=None) with an explicit span_name.
def test_run_single_turn_method_entry():
    entry = next(
        m for m in methods.AGENTS_METHODS
        if m.get("method") == "run_single_turn"
    )
    assert entry["package"] == "agents.run"
    assert entry["object"] is None
    assert entry["span_name"] == "agents.run.run_single_turn"
    assert entry["output_processor"] is AGENT


# Runner.run's turn is skipped only while the agentic.turn scope is open; invocations never are.
def test_runner_run_skips_turn_when_turn_in_progress():
    handler = AgentsSpanHandler()
    turn_wrap = {"output_processor": AGENT_REQUEST}          # Runner.run -> agentic.turn
    invocation_wrap = {"output_processor": AGENT}            # run_single_turn -> invocation

    assert AGENT_REQUEST["type"] == SPAN_TYPES.AGENTIC_REQUEST
    assert AGENT["type"] == SPAN_TYPES.AGENTIC_INVOCATION

    # No turn open -> Runner.run is NOT skipped (canonical single-run app, golden case)
    assert handler.skip_span(turn_wrap, None, None, (), {}) is False

    token = start_scope(AGENT_TURN_SCOPE)
    try:
        # Turn already open -> the redundant Runner.run turn span is skipped
        assert handler.skip_span(turn_wrap, None, None, (), {}) is True
        # ...but run_single_turn invocations are still recorded
        assert handler.skip_span(invocation_wrap, None, None, (), {}) is False
    finally:
        stop_scope(token)

    # Scope closed -> back to creating the turn
    assert handler.skip_span(turn_wrap, None, None, (), {}) is False
