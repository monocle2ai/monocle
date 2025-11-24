from types import SimpleNamespace

from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION
import monocle_apptrace.instrumentation.metamodel.openai.openai_processor as processor

RunnerMock = type("Runner", (), {})


def test_openai_agents_handler_sets_session_scope(monkeypatch):
    handler = processor.OpenAIAgentsSpanHandler()
    session = SimpleNamespace(session_id="sess-openai")

    scope_calls = []
    monkeypatch.setattr(
        processor,
        "set_scope",
        lambda key, value: scope_calls.append((key, value)) or "scope-token",
    )
    remove_calls = []
    monkeypatch.setattr(processor, "remove_scope", lambda token: remove_calls.append(token))
    monkeypatch.setattr(processor, "extract_session_id_from_agents", lambda kwargs: kwargs["session"].session_id)

    token, _ = handler.pre_tracing({}, None, RunnerMock(), (), {"session": session})

    assert token == "scope-token"
    assert scope_calls == [(AGENT_SESSION, "sess-openai")]

    handler.post_tracing({}, None, RunnerMock(), (), {"session": session}, None, token)
    assert remove_calls == ["scope-token"]


def test_openai_agents_handler_ignores_non_runner_instances(monkeypatch):
    handler = processor.OpenAIAgentsSpanHandler()
    monkeypatch.setattr(processor, "set_scope", lambda key, value: (_ for _ in ()).throw(RuntimeError("should not be called")))

    token, _ = handler.pre_tracing({}, None, object(), (), {"session": object()})
    assert token is None

