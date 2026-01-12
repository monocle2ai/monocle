from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION
from monocle_apptrace.instrumentation.common.utils import get_scopes
from monocle_apptrace.instrumentation.metamodel.strands.strands_processor import (
    StrandsSpanHandler,
)


class Agent:
    """Minimal stand-in for strands.Agent that exposes a session_manager."""

    def __init__(self, session_manager=None):
        self._session_manager = session_manager
        self.name = "travel_agent"
        self.description = "Travel booking agent"


class SessionManager:
    def __init__(self, session_id=None):
        self.session_id = session_id


def test_strands_handler_sets_session_scope():
    handler = StrandsSpanHandler()
    agent = Agent(SessionManager(session_id="session-xyz"))
    previous_scope = get_scopes().get(AGENT_SESSION)

    token, _ = handler.pre_tracing({}, None, agent, (), {})

    assert token is not None, "Expected set_scope token when session_id is present"
    assert (
        get_scopes().get(AGENT_SESSION) == "session-xyz"
    ), "agentic.session scope should match the session manager id"
    
    # Manually cleanup since there's no post_tracing
    from monocle_apptrace.instrumentation.common.utils import remove_scope
    if token:
        remove_scope(token)

    assert (
        get_scopes().get(AGENT_SESSION) == previous_scope
    ), "agentic.session scope should be restored after cleanup"


def test_strands_handler_ignores_missing_session():
    handler = StrandsSpanHandler()
    agent = Agent()  # no session_manager
    previous_scope = get_scopes().get(AGENT_SESSION)

    token, _ = handler.pre_tracing({}, None, agent, (), {})

    assert token is None, "Token should stay None when no session_id is available"
    assert (
        get_scopes().get(AGENT_SESSION) == previous_scope
    ), "agentic.session scope should remain unchanged when no session exists"

