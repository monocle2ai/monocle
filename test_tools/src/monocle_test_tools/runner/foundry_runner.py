import asyncio
import logging
import os
import re
from typing import Any, Optional, Union

import requests

from monocle_test_tools.runner.agent_runner import AgentRunner

logger = logging.getLogger(__name__)

# Okahu holds the deployed agent's spans; the trace is looked up by id, not by a fact.
REMOTE_TRACE_SOURCE = "okahu"
REMOTE_TRACE_FACT = "trace"
# Workflow the deployed agent exports under, when not passed to the constructor.
FOUNDRY_TRACE_WORKFLOW_ENV = "FOUNDRY_TRACE_WORKFLOW"

# Entra scope for the Foundry project data plane.
FOUNDRY_SCOPE = "https://ai.azure.com/.default"

# The hosted-agents surface is preview-gated; this is the opt-in azure-ai-projects
# sends for the agents operations.
FOUNDRY_FEATURES_HEADER = "Foundry-Features"
FOUNDRY_FEATURES_VALUE = (
    "WorkflowAgents=V1Preview,ExternalAgents=V1Preview,"
    "DraftAgents=V1Preview,AgentsOptimization=V2Preview"
)

# Required on the agent endpoint route, and rejected on the model-deployment
# route, so it is added only when the caller left it off.
DEFAULT_API_VERSION = "v1"

# The response carries the trace id under this header. Something on the path
# duplicates it as "<id>,<id>", so only the first value is the id.
TRACE_ID_HEADER = "x-request-id"
SESSION_ID_HEADER = "x-agent-session-id"

# .../agents/<name>/endpoint/... — the agent addresses itself by name in the body,
# so the name is read back off the URL rather than asked for twice.
_AGENT_NAME_RE = re.compile(r"/agents/([^/]+)/endpoint\b")


class FoundryRunner(AgentRunner):
    """Invokes an agent already deployed to Azure AI Foundry.

    ``root_agent`` is the agent's Responses URL, shaped
    ``{project}/agents/{agent}/endpoint/protocols/openai/responses`` — not the
    project's ``/openai/v1/responses``, which resolves model deployments.

    Spans come from the deployed agent's own instrumentation and are retrieved
    from Okahu by the trace id in ``x-request-id``. That needs the workflow name
    it exports under (``trace_workflow_name`` or ``FOUNDRY_TRACE_WORKFLOW``);
    without it retrieval is skipped and only the response can be asserted on.
    """

    def __init__(self, credential: Any = None, trace_workflow_name: Optional[str] = None):
        """A ``DefaultAzureCredential`` is built lazily when none is passed, so
        ``az login`` and CI workload identity both work unchanged."""
        self._credential = credential
        self._trace_workflow_name = trace_workflow_name or os.environ.get(
            FOUNDRY_TRACE_WORKFLOW_ENV)
        self._last_trace_id: Optional[str] = None
        self._last_session_id: Optional[str] = None

    def _get_token(self) -> str:
        """Return a bearer token for the Foundry data plane."""
        if self._credential is None:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as e:
                raise ImportError(
                    "azure-identity is required to use the Foundry runner. "
                    "Install it with `pip install azure-identity`."
                ) from e
            self._credential = DefaultAzureCredential()
        return self._credential.get_token(FOUNDRY_SCOPE).token

    @staticmethod
    def _with_api_version(url: str) -> str:
        """Add ``api-version``; the agent route rejects the request without it."""
        if "api-version=" in url:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}api-version={DEFAULT_API_VERSION}"

    @staticmethod
    def _agent_name_from_url(url: str) -> Optional[str]:
        match = _AGENT_NAME_RE.search(url)
        return match.group(1) if match else None

    @staticmethod
    def _build_payload(test_message: Union[str, dict, Any], model: Optional[str]) -> dict:
        """Encode the message as a non-streaming Responses body.

        A dict is sent verbatim. Streaming is off so assertions read one JSON body.
        """
        if isinstance(test_message, dict):
            return test_message
        body: dict = {"input": test_message, "stream": False}
        if model:
            body["model"] = model
        return body

    @staticmethod
    def _first_header_value(value: Optional[str]) -> Optional[str]:
        """First value of a comma-duplicated header; Foundry returns ``<id>,<id>``."""
        if not value:
            return None
        return value.split(",")[0].strip() or None

    def get_last_trace_id(self) -> Optional[str]:
        """Trace id the deployed agent reported for the last call."""
        return self._last_trace_id

    def get_last_session_id(self) -> Optional[str]:
        """Foundry session id for the last call, for correlation only.

        Not the trace id; Foundry mints a new one unless the call is chained
        with ``previous_response_id``.
        """
        return self._last_session_id

    async def run_agent_async(self, root_agent: str, *args, timeout: int = 180,
                              model: Optional[str] = None, headers: Optional[dict] = None,
                              **kwargs) -> Any:
        """Invoke the deployed agent over the Responses protocol.

        The first positional arg is the message; ``model`` defaults to the agent
        name in the URL, ``headers`` merge over the defaults, and the rest is
        passed to ``requests.post``. Returns the ``requests.Response``.
        """
        if root_agent is None or not isinstance(root_agent, str):
            raise ValueError(
                "For FoundryRunner, root_agent must be the deployed agent's Responses URL string."
            )

        test_message = args[0] if args else None
        url = self._with_api_version(root_agent)
        body = self._build_payload(test_message, model or self._agent_name_from_url(url))

        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_token()}",
            FOUNDRY_FEATURES_HEADER: FOUNDRY_FEATURES_VALUE,
        }
        if headers:
            request_headers.update(headers)

        # Identifiers belong to one invocation; a reused runner must not report
        # the previous call's.
        self._last_trace_id = None
        self._last_session_id = None

        response = requests.post(url, json=body, headers=request_headers,
                                 timeout=timeout, **kwargs)
        logger.debug(f"Foundry response status={response.status_code}")

        # Read before raise_for_status so a failing call still says which trace
        # to go and look at.
        self._last_trace_id = self._first_header_value(response.headers.get(TRACE_ID_HEADER))
        self._last_session_id = self._first_header_value(
            response.headers.get(SESSION_ID_HEADER))

        response.raise_for_status()
        return response

    def get_remote_traces_source(self) -> Optional[str]:
        """Report a source only once retrieval can succeed, so an unconfigured
        runner skips it instead of importing nothing."""
        if self._trace_workflow_name and self._last_trace_id:
            return REMOTE_TRACE_SOURCE
        return None

    def get_remote_trace_query(self) -> dict:
        """Identify the deployed agent's spans by the trace id it reported.

        The agent exports under the same trace id Foundry returns in
        ``x-request-id``, an id this process never used — so the spans found can
        only be the deployed agent's.
        """
        if not (self._trace_workflow_name and self._last_trace_id):
            return {}
        return {
            "id": self._last_trace_id,
            "fact_name": REMOTE_TRACE_FACT,
            "workflow_name": self._trace_workflow_name,
        }

    def run_agent(self, root_agent: str, *args, **kwargs) -> Any:
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
