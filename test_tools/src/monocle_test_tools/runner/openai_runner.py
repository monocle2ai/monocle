import asyncio
from typing import Any, Optional, Union
import logging
from agents import Runner, SQLiteSession
logger = logging.getLogger(__name__)
from monocle_test_tools.runner.agent_runner import AgentRunner

class OpenAIRunner(AgentRunner):
    """Runner for OpenAI Agents SDK agents."""

    def __init__(self):
        self._sessions: dict[str, SQLiteSession] = {}

    def _get_session(self, session_id: str = None) -> Optional[SQLiteSession]:
        """Return the conversation memory for a session id, creating it once.

        Later turns with the same session id reuse the same in-memory session, so
        the agent sees the earlier turns of the conversation. No session id means
        a single stateless run.
        """
        if session_id is None:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            session = SQLiteSession(session_id)
            self._sessions[session_id] = session
        return session

    async def run_agent_async(self, root_agent, request: Union[str, Any], session_id: str = None):
        result = await Runner.run(
            root_agent,
            request,
            session=self._get_session(session_id),
        )
        logger.debug(f"OpenAI agent result: {result.final_output}")
        return result.final_output

    def run_agent(self, root_agent, request: Union[str, Any], session_id: str = None):
        return asyncio.run(self.run_agent_async(root_agent, request, session_id=session_id))

    async def end_session(self, session_id: str = None) -> None:
        """Drop the cached session memory once a multi-turn run is complete."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()
