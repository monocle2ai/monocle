"""Runner for an agent that speaks the A2A protocol.

A remote runner: `root_agent` is the agent's base URL, not an agent object, and
the agent runs in another process. The test drives it with the A2A SDK, which
Monocle instruments, so the call itself produces a client-side span.

The agent's own spans are collected afterwards. Set the workflow the agent
exports under and the runner pulls them from Okahu; leave it unset and it
fetches nothing, so the test can import the trace file the agent writes. Both
work because the call carries the trace context, so the agent's spans share the
test's trace id.
"""
import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from monocle_test_tools.runner.agent_runner import AgentRunner

logger = logging.getLogger(__name__)

REMOTE_TRACE_SOURCE = "okahu"
# Workflow the A2A agent exports under, when not passed to the constructor.
A2A_TRACE_WORKFLOW_ENV = "A2A_TRACE_WORKFLOW"

DEFAULT_TIMEOUT_SECONDS = 60

# Task states that accept another message. A task in any other state is
# finished, and sending to it fails, so only these carry into the next turn.
CONTINUABLE_TASK_STATES = frozenset(
    {"submitted", "working", "input-required", "auth-required"})


class A2ARunner(AgentRunner):
    """Runner that sends A2A messages to a remote agent."""

    def __init__(self, agent_card: Any = None, agent_card_path: Optional[str] = None,
                 trace_workflow_name: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 transport: Any = None):
        """
        Args:
            agent_card: Ready-made `AgentCard`. Fetched from the agent if omitted.
            agent_card_path: Non-default path to fetch the card from.
            trace_workflow_name: Workflow the agent exports its spans under,
                needed to find them in Okahu since it is not the test's
                workflow. Falls back to `A2A_TRACE_WORKFLOW`.
            timeout: httpx timeout for the card fetch and the message.
            transport: httpx transport to wrap, for tests standing in for the network.
        """
        self._agent_card = agent_card
        self._agent_card_path = agent_card_path
        self._transport = transport
        self._trace_workflow_name = trace_workflow_name or os.environ.get(
            A2A_TRACE_WORKFLOW_ENV)
        self._timeout = timeout
        # A2A continuation ids per session, so a turn lands in the task and
        # context the turn before it created.
        self._sessions: dict = {}

    @staticmethod
    def _build_params(message: Any, task_id: Optional[str],
                      context_id: Optional[str]) -> dict:
        """Build the `MessageSendParams` body for one turn.

        A string becomes a single text part. A dict holding "message" is used
        as-is, so any shape the protocol allows can be sent; any other dict is
        taken as the message itself. Continuation ids fill in only where the
        caller left them out.
        """
        if isinstance(message, dict):
            params = dict(message) if "message" in message else {"message": dict(message)}
        else:
            params = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": str(message)}],
                }
            }
        msg = dict(params["message"])
        msg.setdefault("messageId", uuid4().hex)
        if task_id and not msg.get("taskId"):
            msg["taskId"] = task_id
        if context_id and not msg.get("contextId"):
            msg["contextId"] = context_id
        params["message"] = msg
        return params

    def _remember_turn(self, session_key: Any, response: Any) -> None:
        """Keep the task and context to continue on the next turn.

        A Task result carries its own id and a context id; a Message result
        carries both as fields. Read defensively: an error response has
        neither, and none of the fields are required. The task is kept only
        while it still accepts messages; the context always is.
        """
        result = getattr(getattr(response, "root", None), "result", None)
        if result is None:
            return
        if getattr(result, "kind", None) == "task":
            state = getattr(getattr(result, "status", None), "state", None)
            task_id = (getattr(result, "id", None)
                       if getattr(state, "value", state) in CONTINUABLE_TASK_STATES else None)
        else:
            task_id = getattr(result, "task_id", None)
        context_id = getattr(result, "context_id", None)
        if task_id or context_id:
            self._sessions[session_key] = {"task_id": task_id, "context_id": context_id}

    async def run_agent_async(self, root_agent: Any, *args, session_id: str = None,
                              task_id: str = None, context_id: str = None,
                              http_kwargs: dict = None, **kwargs) -> Any:
        """Send one message to the agent and return its `SendMessageResponse`.

        Args:
            root_agent: The agent's base URL, or an `AgentCard`.
            session_id: Groups the turns of a multi-turn run.
            task_id, context_id: Continue this task or context instead of the
                one held for the session.
            http_kwargs: Passed to `A2AClient.send_message`.
        """
        message = args[0] if args else kwargs.pop("message", None)
        if message is None:
            raise ValueError("For A2ARunner, a message to send is required.")

        a2a_client, a2a_types = _import_a2a()
        from monocle_test_tools.a2a_transport import make_traced_httpx_client

        remembered = self._sessions.get(session_id, {})
        params = self._build_params(
            message,
            task_id or remembered.get("task_id"),
            context_id or remembered.get("context_id"),
        )

        async with make_traced_httpx_client(transport=self._transport,
                                            timeout=self._timeout) as httpx_client:
            agent_card = await self._resolve_agent_card(a2a_client, httpx_client, root_agent)
            client = a2a_client.A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            request = a2a_types.SendMessageRequest(
                id=str(uuid4()),
                params=a2a_types.MessageSendParams(**params),
            )
            response = await client.send_message(request, http_kwargs=http_kwargs)

        self._remember_turn(session_id, response)
        return response

    async def _resolve_agent_card(self, a2a_client, httpx_client, root_agent: Any) -> Any:
        """Return the card to talk to: the configured one, one passed in, or the agent's."""
        if self._agent_card is not None:
            return self._agent_card
        if not isinstance(root_agent, str):
            return root_agent
        resolver_kwargs = {"agent_card_path": self._agent_card_path} if self._agent_card_path else {}
        resolver = a2a_client.A2ACardResolver(
            httpx_client=httpx_client, base_url=root_agent, **resolver_kwargs)
        return await resolver.get_agent_card()

    def run_agent(self, root_agent: Any, *args, **kwargs) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run,
                                     self.run_agent_async(root_agent, *args, **kwargs))
                return future.result()
        return asyncio.run(self.run_agent_async(root_agent, *args, **kwargs))

    async def end_session(self, session_id: str = None) -> None:
        """Drop the continuation ids held for a finished session."""
        self._sessions.pop(session_id, None)

    def get_remote_traces_source(self) -> Optional[str]:
        """Pull from Okahu only when the agent's workflow is known.

        Without it there is nothing to ask Okahu for, so the runner fetches
        nothing and the test reads the agent's trace file instead.
        """
        return REMOTE_TRACE_SOURCE if self._trace_workflow_name else None

    def get_remote_trace_query(self) -> dict:
        """Name the workflow the agent exports under.

        The agent's spans are in the test's trace, so the validator resolves
        the trace id from the local spans and only the workflow is needed here.
        """
        if not self._trace_workflow_name:
            return {}
        return {"workflow_name": self._trace_workflow_name}


def _import_a2a():
    """Import the A2A SDK lazily, so it stays an optional dependency."""
    try:
        from a2a import client as a2a_client
        from a2a import types as a2a_types
    except ImportError as e:
        raise ImportError(
            "The a2a SDK is required to use the A2A runner. "
            "Install it with `pip install monocle_test_tools[a2a]`."
        ) from e
    return a2a_client, a2a_types
