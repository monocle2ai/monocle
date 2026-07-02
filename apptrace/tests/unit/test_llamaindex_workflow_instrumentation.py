"""Unit tests for the LlamaIndex Workflow / structured-LLM instrumentation fixes.

Covers:
  * base ``Workflow.run`` -> agentic.turn handler (session anchoring + skip rules)
  * deferred span completion via ``process_workflow_handler``
  * I/O extraction fixes: keyword ``messages``, non-scalar tool args,
    Pydantic/dict/list tool outputs, workflow input/output.
"""
import asyncio

import monocle_apptrace.instrumentation.metamodel.llamaindex.llamaindex_processor as processor
from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION, AGENT_REQUEST_SPAN_NAME
from monocle_apptrace.instrumentation.metamodel.llamaindex import _helper
from monocle_apptrace.instrumentation.metamodel.llamaindex.entities.agent import (
    WORKFLOW,
    process_workflow_handler,
)


# --------------------------------------------------------------------------- #
# Workflow.run handler
# --------------------------------------------------------------------------- #
class _DummyWorkflow:
    """Stand-in for a plain (non-agent) llama_index Workflow."""


def test_workflow_handler_pre_tracing_sets_session_scope(monkeypatch):
    scope_calls = []
    monkeypatch.setattr(processor, "set_scope", lambda key, value: scope_calls.append((key, value)) or "scope-token")
    handler = processor.LlamaIndexWorkflowHandler()

    token, alt = handler.pre_tracing({}, None, _DummyWorkflow(), (), {})

    assert token == "scope-token"
    assert alt is None
    # No session id available -> set_scope called with None (auto-generated downstream).
    assert scope_calls == [(AGENT_SESSION, None)]


def test_workflow_handler_skips_when_turn_scope_already_set(monkeypatch):
    monkeypatch.setattr(processor, "is_scope_set", lambda name: name == AGENT_REQUEST_SPAN_NAME)
    handler = processor.LlamaIndexWorkflowHandler()
    assert handler.skip_span({}, None, _DummyWorkflow(), (), {}) is True


def test_workflow_handler_does_not_skip_plain_workflow(monkeypatch):
    monkeypatch.setattr(processor, "is_scope_set", lambda name: False)
    handler = processor.LlamaIndexWorkflowHandler()
    assert handler.skip_span({}, None, _DummyWorkflow(), (), {}) is False


# --------------------------------------------------------------------------- #
# Deferred span completion
# --------------------------------------------------------------------------- #
def test_process_workflow_handler_defers_then_closes_on_completion():
    loop = asyncio.new_event_loop()
    try:
        fut = loop.create_future()
        closed = []
        returned = process_workflow_handler({}, fut, lambda r: closed.append(r))

        assert returned is fut          # handler returned unchanged
        assert closed == []             # span NOT closed before completion

        fut.set_result({"answer": 38})
        loop.run_until_complete(asyncio.sleep(0))  # flush done callbacks

        assert closed == [{"answer": 38}]
    finally:
        loop.close()


def test_process_workflow_handler_closes_immediately_without_future():
    closed = []
    returned = process_workflow_handler({}, None, lambda r: closed.append(r))
    assert returned is None
    assert closed == [None]


def test_workflow_entity_is_deferred_turn():
    assert WORKFLOW["type"] == AGENT_REQUEST_SPAN_NAME  # agentic.turn
    assert WORKFLOW["is_auto_close"]({}) is False
    assert WORKFLOW["response_processor"] is process_workflow_handler


# --------------------------------------------------------------------------- #
# I/O extraction fixes
# --------------------------------------------------------------------------- #
class _Msg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


def test_extract_messages_reads_keyword_messages_when_args_empty():
    # Structured-LLM path calls achat(messages=[...]) -> args empty, messages in kwargs.
    kwargs = {"messages": [_Msg("user", "Query: how many movies? Answer:")]}
    out = _helper.extract_messages(kwargs)
    assert out and any("how many movies" in m for m in out)


class _ToolOutput:
    def __init__(self, raw_output, content=""):
        self.raw_output = raw_output
        self.content = content


class _Pydanticish:
    def __init__(self, data):
        self._data = data

    def model_dump_json(self):
        import json
        return json.dumps(self._data)


def test_extract_tool_response_serializes_pydantic_raw_output():
    resp = _ToolOutput(_Pydanticish({"decision": "movie"}), content="decision='movie'")
    assert _helper.extract_tool_response(resp) == '{"decision": "movie"}'


def test_extract_tool_response_falls_back_to_content_when_raw_output_empty():
    resp = _ToolOutput(None, content="fallback text")
    assert _helper.extract_tool_response(resp) == "fallback text"


def test_extract_tool_args_includes_non_scalar_values():
    arguments = {"args": (), "kwargs": {"plan": [["q1"], ["q2"]], "n": 3}}
    out = _helper.extract_tool_args(arguments)
    assert '"plan"' in out and "q1" in out and '"n": 3' in out


def test_extract_workflow_input_prefers_input_kwarg():
    assert _helper.extract_workflow_input((), {"input": "How many movies?"}) == "How many movies?"


def test_extract_workflow_output_serializes_dict_result():
    out = _helper.extract_workflow_output({"result": {"answer": "38"}, "exception": None})
    assert out == '{"answer": "38"}'


class _MemMsg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class _Memory:
    def __init__(self, msgs):
        self._msgs = msgs

    def get_all(self):
        return self._msgs


def test_extract_agent_input_falls_back_to_memory_user_message():
    # Workflow-agent finalize(ctx, output, memory): the query lives in memory, not args.
    mem = _Memory([_MemMsg("user", "Who wrote X?"), _MemMsg("assistant", "...")])
    out = _helper.extract_agent_input((object(), object(), mem), {})
    assert out == ["Who wrote X?"]


def test_extract_agent_input_memory_via_kwargs():
    mem = _Memory([_MemMsg("user", "latest query")])
    out = _helper.extract_agent_input((), {"memory": mem})
    assert out == ["latest query"]


def test_extract_agent_input_prefers_positional_when_present():
    out = _helper.extract_agent_input(("hello",), {})
    assert out == ["hello"]
