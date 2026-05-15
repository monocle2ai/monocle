import asyncio
import logging
from typing import Union, Any, Optional
import requests
from monocle_test_tools.runner.agent_runner import AgentRunner
from monocle_apptrace.instrumentation.metamodel.requests._helper import RequestSpanHandler

logger = logging.getLogger(__name__)

class HttpRunner(AgentRunner):
    async def run_agent_async(self, root_agent: str, *args, **kwargs) -> str:
        """Run the given agent with provided root_agent URL, args and kwargs."""
        """ Arguments:
            root_agent: The URL to which the HTTP request will be made. This is expected to be a string URL. For HttpRunner, root_agent is used as the target URL for the HTTP request.
            session_id: Optional session ID for tracing purposes.
            *args: Positional arguments that may contain request data or parameters.
            **kwargs: Keyword arguments that may contain request details like method, headers, body, etc.
        """
        # For HttpRunner, we expect the root_agent to be a callable that takes a request dict
        if root_agent is None or not isinstance(root_agent, str):
            raise ValueError("For HttpRunner, root_agent is not expected. Please pass the HTTP request details in args and kwargs.")

        # Use request library to make the HTTP request based on the details in args and kwargs
        # method = kwargs.get("method", "GET")
        # url = root_agent
        # headers = kwargs.get("headers", {})
        # data = kwargs.get("data", None) or kwargs.get("json", None) or kwargs.get("body", None) or (args[0] if len(args) > 0 else None) or ""
        try:
            RequestSpanHandler.set_trace_all_urls_for_test(True)  # Ensure all requests are traced for testing
            kwargs["url"] = root_agent
            response = requests.request(**kwargs)
            logger.debug(f"Received HTTP response with status code: {response.status_code}, body: {response.text}")
            response.raise_for_status()  # Raise an exception for HTTP error codes
            return response
        finally:
            RequestSpanHandler.set_trace_all_urls_for_test(False)  # Reset to default after the request is done

    def run_agent(self, root_agent: str, *args, **kwargs) -> str:
        import asyncio
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
    
    def get_remote_traces_source(self) -> str:
        """HttpRunner may have remote traces if the HTTP request triggers spans that are exported to a remote backend. This can be overridden if the runner needs to fetch traces in a specific way."""
        return "okahu"