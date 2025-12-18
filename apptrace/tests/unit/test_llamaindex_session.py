import types

from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION
import monocle_apptrace.instrumentation.metamodel.llamaindex.llamaindex_processor as processor


class DummyMemory:
    def __init__(self, session_id=None):
        self.session_id = session_id


class DummyAgent:
    def __init__(self, name="travel_agent"):
        self.name = name


def _patch_llamaindex_context(monkeypatch):
    monkeypatch.setattr(processor, "get_current", lambda: "ctx")
    monkeypatch.setattr(processor, "set_current_agent", lambda name: None)
    monkeypatch.setattr(processor, "get_name", lambda instance: instance.name)

    def fake_set_value(key, value, ctx):
        return (ctx, key, value)

    monkeypatch.setattr(processor, "set_value", fake_set_value)

    attach_tokens = []

    def fake_attach(ctx):
        attach_tokens.append(ctx)
        return "context-token"

    detach_tokens = []

    def fake_detach(token):
        detach_tokens.append(token)

    monkeypatch.setattr(processor, "attach", fake_attach)
    monkeypatch.setattr(processor, "detach", fake_detach)

    scope_calls = []

    def fake_set_scope(key, value, ctx):
        scope_calls.append((key, value))
        return "scope-token"

    monkeypatch.setattr(processor, "set_scope", fake_set_scope)

    return scope_calls, detach_tokens


def test_llamaindex_agent_handler_sets_session_scope(monkeypatch):
    scope_calls, detach_calls = _patch_llamaindex_context(monkeypatch)
    handler = processor.LlamaIndexAgentHandler()

    session_token, _ = handler.pre_tracing(
        {},
        None,
        DummyAgent(),
        (),
        {"memory": DummyMemory(session_id="session-123")},
    )

    assert session_token == "scope-token"
    assert scope_calls == [(AGENT_SESSION, "session-123")]


def test_llamaindex_agent_handler_skips_scope_when_session_missing(monkeypatch):
    scope_calls, detach_calls = _patch_llamaindex_context(monkeypatch)
    handler = processor.LlamaIndexAgentHandler()

    session_token, _ = handler.pre_tracing(
        {},
        None,
        DummyAgent(),
        (),
        {"memory": DummyMemory(session_id=None)},
    )

    # When no session_id, it returns a context token from attach(), not None
    assert session_token == "context-token"
    assert scope_calls == []  # set_scope should not be called

