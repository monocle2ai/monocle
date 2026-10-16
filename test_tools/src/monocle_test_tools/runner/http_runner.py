import asyncio
import logging
import os
import warnings
from typing import Any, Optional
import requests
from monocle_test_tools.runner.agent_runner import AgentRunner, remote_trace_source
from monocle_test_tools.file_span_loader import JSONSpanLoader
from monocle_apptrace.instrumentation.metamodel.requests._helper import RequestSpanHandler
from monocle_apptrace.instrumentation.common.constants import (
    MONOCLE_TRACE_RETRIEVAL_KEY_ENV,
    TRACE_RETURN_REQUEST_HEADER,
)

logger = logging.getLogger(__name__)


class _BaseHttpRunner(AgentRunner):
    async def run_agent_async(self, root_agent: str, *args, **kwargs) -> Any:
        """Run the given agent with provided root_agent URL, args and kwargs."""
        """ Arguments:
            root_agent: The URL to which the HTTP request will be made. This is expected to be a string URL. For HttpRunner, root_agent is used as the target URL for the HTTP request.
            session_id: Optional session ID for tracing purposes.
            *args: Positional arguments that may contain request data or parameters.
            **kwargs: Keyword arguments that may contain request details like method, headers, body, etc.
        """
        if root_agent is None or not isinstance(root_agent, str):
            raise ValueError("For HttpRunner, root_agent must be the target URL string.")
        try:
            RequestSpanHandler.set_trace_all_urls_for_test(True)  # Ensure all requests are traced for testing
            kwargs["url"] = root_agent
            response = requests.request(**kwargs)
            logger.debug(f"HTTP response status={response.status_code}")
            response.raise_for_status()  # Raise an exception for HTTP error codes
            return response
        finally:
            RequestSpanHandler.set_trace_all_urls_for_test(False)  # Reset to default after the request is done

    def run_agent(self, root_agent: str, *args, **kwargs) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.run_agent_async(root_agent, *args, **kwargs))
                return future.result()
        return asyncio.run(self.run_agent_async(root_agent, *args, **kwargs))


class HttpRemoteTraceRunner(_BaseHttpRunner):
    """HTTP runner that fetches server-side traces from a trace source after the request.

    The server exports its spans to that source; ``trace_source`` names it
    (falls back to ``MONOCLE_REMOTE_TRACE_SOURCE``).
    """

    def __init__(self, trace_source: Optional[str] = None):
        super().__init__()
        self._trace_source = trace_source

    def get_remote_traces_source(self) -> Optional[str]:
        return remote_trace_source(self._trace_source)


class HttpOkahuRunner(HttpRemoteTraceRunner):
    """Old name, kept for one release: HttpRemoteTraceRunner(trace_source="okahu")."""

    def __init__(self):
        warnings.warn("HttpOkahuRunner is deprecated; use HttpRemoteTraceRunner(trace_source=\"okahu\")",
                      DeprecationWarning, stacklevel=2)
        super().__init__(trace_source="okahu")


class HttpRunner(_BaseHttpRunner):
    """HTTP runner that reads server-side spans piggybacked on the HTTP response."""

    def __init__(self):
        super().__init__()
        self._remote_spans = []

    async def run_agent_async(self, root_agent: str, *args, **kwargs) -> Any:
        self._maybe_inject_retrieval_key(kwargs)
        response = await super().run_agent_async(root_agent, *args, **kwargs)
        self._capture_remote_spans(response)
        return response

    def _maybe_inject_retrieval_key(self, kwargs) -> None:
        """Add the trace-retrieval key header from MONOCLE_TRACE_RETRIEVAL_KEY,
        unless the caller already supplied the header."""
        key = os.environ.get(MONOCLE_TRACE_RETRIEVAL_KEY_ENV)
        if not key:
            return
        headers = dict(kwargs.get("headers") or {})
        if any(str(k).lower() == TRACE_RETURN_REQUEST_HEADER for k in headers):
            return  # caller-supplied header wins
        headers[TRACE_RETURN_REQUEST_HEADER] = key
        kwargs["headers"] = headers

    def _capture_remote_spans(self, response) -> None:
        raw = getattr(response, "_monocle_remote_spans", None)
        if raw:
            try:
                self._remote_spans = JSONSpanLoader.from_json_str(raw)
            except Exception as e:
                logger.warning(f"Failed to deserialize piggybacked spans: {e}")
                self._remote_spans = []

    def get_remote_spans(self) -> list:
        return self._remote_spans

    def get_remote_traces_source(self):
        return None
