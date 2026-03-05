import asyncio
import inspect
import logging
from typing import Any, Union

from monocle_test_tools.runner.agent_runner import AgentRunner

logger = logging.getLogger(__name__)


class MSAgentRunner(AgentRunner):
    """Runner for Microsoft Agent Framework agents."""

    async def run_agent_async(self, root_agent, request: Union[str, dict, Any], session_id: str = None):
        """Execute an MS Agent asynchronously.

        Args:
            root_agent: The MS Agent instance (for example ChatAgent/as_agent result).
            request: Input message or request payload.
            session_id: Optional service thread id for session continuity.
        """
        thread = None
        if hasattr(root_agent, "get_new_thread"):
            try:
                if session_id:
                    thread = root_agent.get_new_thread(service_thread_id=session_id)
                else:
                    thread = root_agent.get_new_thread()
            except Exception as exc:
                logger.debug(f"Unable to create/reuse MS Agent thread: {exc}")

        if hasattr(root_agent, "run"):
            if isinstance(request, dict):
                kwargs = request.copy()
                if thread is not None and "thread" not in kwargs:
                    kwargs["thread"] = thread
                result = await root_agent.run(**kwargs)
            else:
                if thread is not None:
                    result = await root_agent.run(request, thread=thread)
                else:
                    result = await root_agent.run(request)
        else:
            raise AttributeError(
                f"Agent {type(root_agent).__name__} does not have async 'run' method"
            )

        if hasattr(result, "text") and result.text is not None:
            return str(result.text)
        if hasattr(result, "output_text") and result.output_text is not None:
            return str(result.output_text)
        if hasattr(result, "content") and result.content is not None:
            return str(result.content)
        return str(result) if result is not None else None

    def run_agent(self, root_agent, request: Union[str, dict, Any], session_id: str = None):
        """Execute an MS Agent in sync context."""
        async_runner = self.run_agent_async(root_agent, request, session_id=session_id)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(async_runner)

        if loop.is_running():
            raise RuntimeError(
                "run_agent called inside an active event loop. Use run_agent_async instead."
            )
        return loop.run_until_complete(async_runner)
