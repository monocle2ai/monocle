from types import SimpleNamespace

from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION
from monocle_apptrace.instrumentation.common.utils import get_scopes
import monocle_apptrace.instrumentation.metamodel.openai.openai_processor as processor


class Runner:

    pass

def test_openai_agents_handler_sets_session_scope():
    handler = processor.OpenAIAgentsSpanHandler()
    session = SimpleNamespace(session_id="abc_123")

    previous_scope = get_scopes().get(AGENT_SESSION)

    token, _ = handler.pre_tracing({}, None, Runner(), (), {"session": session})

    try:
        assert token is not None, "Expected token when session_id provided"
        assert (
            get_scopes().get(AGENT_SESSION) == "abc_123"
        ), "Session scope should match provided session_id"
    finally:
        handler.post_tracing({}, None, Runner(), (), {"session": session}, None, token)

    assert (
        get_scopes().get(AGENT_SESSION) == previous_scope
    ), "Session scope should be restored after cleanup"


def test_openai_agents_handler_ignores_non_runner_instances():
    handler = processor.OpenAIAgentsSpanHandler()
    previous_scope = get_scopes().get(AGENT_SESSION)

    token, _ = handler.pre_tracing({}, None, object(), (), {"session": object()})
    handler.post_tracing({}, None, object(), (), {"session": object()}, None, token)

    assert token is None, "Token should stay None when instance isn't a Runner"
    assert (
        get_scopes().get(AGENT_SESSION) == previous_scope
    ), "Session scope should remain unchanged for non-runner instances"

