from abc import abstractmethod
from typing import Any

class AgentRunner:
    @abstractmethod
    async def run_agent_async(self, root_agent,*args, **kwargs) -> Any:
        raise NotImplementedError("This is a placeholder function. Please implement the function in your test setup.")

    @abstractmethod
    def run_agent(self, root_agent, *args, **kwargs) -> Any:
        raise NotImplementedError("This is a placeholder function. Please implement the function in your test setup.")
    
    def get_remote_traces_source(self) -> str:
        """Check if the runner has remote traces. This can be overridden if the runner needs to fetch traces in a specific way."""
        return None

    async def end_session(self, session_id: str = None) -> None:
        """Release any resources held for a live multi-turn session.

        Runners that keep per-session state alive across turns (for example the
        ADK runner, which caches an in-memory session service so the agent's
        memory persists) should override this to drop that state once the
        multi-turn run finishes. Runners that delegate session continuity to
        their own framework (LangGraph thread_id, Strands FileSessionManager,
        LlamaIndex chat store, MS Agent thread) need no cleanup, so the default
        is a no-op.
        """
        return None

