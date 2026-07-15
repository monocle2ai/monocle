"""Unit regression tests for three CrewAI coverage gaps surfaced by a 4-agent
sequential demo (Devyan) on CrewAI 1.15.x:

1. Custom tools built with the @tool decorator are Tool(BaseTool) instances whose own
   run() shadows BaseTool.run (and does not call super), so the BaseTool.run wrapper
   never fired and decorator tools produced no agentic.tool.invocation span. Tool.run
   must be registered too.

2. A crew.kickoff() run had no scope.agentic.session anchoring it. CrewAI has no
   thread/session id of its own, so CrewAIAgentHandler must start an auto-generated
   agentic.session scope at the outermost call (and never override a caller's).

3. Task.execute_sync carries the task prompt on the instance, not in kwargs. The first
   task (no prior-task context) therefore had an empty data.input. extract_agent_input
   must fall back to the instance's description.

None of these require a live LLM or network call.
"""
from opentelemetry import baggage
from opentelemetry.context import detach

from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION
from monocle_apptrace.instrumentation.common.utils import set_scope, MONOCLE_SCOPE_NAME_PREFIX
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.crew_ai.methods import CREW_AI_METHODS
from monocle_apptrace.instrumentation.metamodel.crew_ai.crew_ai_processor import CrewAIAgentHandler
from monocle_apptrace.instrumentation.metamodel.crew_ai import _helper


def _by_object_method():
    return {(m["object"], m["method"]): m for m in CREW_AI_METHODS}


# --- Fix 1: decorator-tool run() is instrumented -----------------------------

def test_decorator_tool_run_is_registered():
    entry = _by_object_method().get(("Tool", "run"))
    assert entry is not None, "crewai.tools.base_tool.Tool.run must be wrapped"
    assert entry["package"] == "crewai.tools.base_tool"
    assert entry["wrapper_method"] is task_wrapper
    assert entry["span_handler"] == "crew_ai_tool_handler"
    # Distinct from the BaseTool.run entry (Tool overrides run without calling super).
    assert ("BaseTool", "run") in _by_object_method()


# --- Fix 2: session scope anchored at the outermost call ----------------------

class _FakeCrew:
    role = "CrewAI"  # get_name() uses .role; no .stream attr -> not streaming


def test_agent_handler_starts_session_scope_at_root():
    handler = CrewAIAgentHandler()
    scope_key = f"{MONOCLE_SCOPE_NAME_PREFIX}{AGENT_SESSION}"
    assert baggage.get_baggage(scope_key) is None

    token, wrapper = handler.pre_tracing({"method": "kickoff"}, None, _FakeCrew(), (), {})
    try:
        assert wrapper is not None  # turn (agentic.request) wrapper added at the root
        assert baggage.get_baggage(scope_key) is not None  # session anchored
    finally:
        detach(token)
    assert baggage.get_baggage(scope_key) is None  # scope released on detach


def test_agent_handler_no_extra_wrapper_when_turn_scope_already_set():
    handler = CrewAIAgentHandler()
    turn_token = set_scope("agentic.turn", "turn-1")
    try:
        token, wrapper = handler.pre_tracing({"method": "execute_task"}, None, _FakeCrew(), (), {})
        assert wrapper is None  # nested call: no second turn span
        detach(token)
    finally:
        detach(turn_token)


# --- Fix 3: task prompt (instance.description) becomes data.input -------------

class _FakeTask:
    description = "Provide a high-level solution architecture for the given problem."


def test_extract_agent_input_uses_task_instance_description():
    # Task.execute_sync: prompt lives on the instance, kwargs has no 'task'.
    args = {"instance": _FakeTask(), "args": (), "kwargs": {}}
    assert _helper.extract_agent_input(args) == [_FakeTask.description]


def test_extract_agent_input_includes_instance_description_and_context():
    args = {"instance": _FakeTask(), "args": (), "kwargs": {"context": "prior task output"}}
    result = _helper.extract_agent_input(args)
    assert result[0] == _FakeTask.description
    assert "context: prior task output" in result


def test_extract_agent_input_prefers_kwargs_task_for_agent_execute():
    # Agent.execute_task path is unchanged: kwargs['task'].description wins.
    class _TaskArg:
        description = "the task prompt"
    args = {"instance": object(), "args": (), "kwargs": {"task": _TaskArg()}}
    assert _helper.extract_agent_input(args) == ["the task prompt"]
