import logging
import os
from typing import Any, Dict, List, Optional
import requests
from opentelemetry.sdk.trace import ReadableSpan
from monocle_test_tools.file_span_loader import JSONSpanLoader

logger = logging.getLogger(__name__)


class OkahuSpanLoader:
    """Utility class to load spans from Okahu trace service.

    Uses the Okahu REST API:
        - GET /api/v1/workflows/<wf_name>/traces?duration_fact=<fact>&fact_ids=<id>
          Get traces matching a fact (e.g. ``agentic_session``).
        - GET /api/v1/workflows/<wf_name>/traces/<trace_id>/spans
          Get spans for a trace, optionally filtered by session.

    Base URL defaults to https://api.okahu.co and can be overridden
    with the OKAHU_API_ENDPOINT environment variable.
    """

    # Constants
    AGENT_SESSIONS_SCOPE = "agent_sessions"
    OKAHU_BASE_URL = "https://api.okahu.co"

    @staticmethod
    def _get_api_base(endpoint: Optional[str] = None) -> str:
        """Return the Okahu API base URL (no trailing slash)."""
        return (endpoint or os.environ.get("OKAHU_API_ENDPOINT", OkahuSpanLoader.OKAHU_BASE_URL)).rstrip("/")

    @staticmethod
    def _get_headers(api_key: Optional[str] = None) -> dict:
        """Return common request headers."""
        key = api_key or os.environ.get("OKAHU_API_KEY")
        if not key:
            raise ValueError("OKAHU_API_KEY is not configured. Set the environment variable or pass api_key.")
        return {
            "Content-Type": "application/json",
            "x-api-key": key
        }

    @staticmethod
    def _do_get(url: str, headers: dict, params: Optional[dict] = None,
                timeout: int = 30, context_msg: str = "") -> Any:
        """Execute a GET request with standard error handling."""
        try:
            response = requests.get(url=url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise ConnectionError(f"Okahu request timed out ({context_msg}): {exc}") from exc
        except requests.HTTPError as exc:
            raise
        except requests.RequestException as exc:
            raise ConnectionError(f"Failed to reach Okahu service ({context_msg}): {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise ConnectionError(
                f"Okahu returned invalid JSON ({context_msg}): {response.text}"
            ) from exc

    @staticmethod
    def _unwrap_list(data: Any, wrapper_keys: tuple, context_msg: str = "") -> list:
        """Unwrap a list from a possible dict wrapper."""
        if isinstance(data, dict):
            for key in wrapper_keys:
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise ConnectionError(
                f"Okahu response is a dict but no known list key found ({context_msg}). "
                f"Keys: {list(data.keys())}"
            )
        if isinstance(data, list):
            return data
        raise ConnectionError(
            f"Expected a list from Okahu ({context_msg}), got: {type(data).__name__}"
        )

    # ------------------------------------------------------------------ #
    #  Public helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_trace_ids(
        workflow_name: str,
        fact_name: str,
        fact_id: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> List[str]:
        """Fetch trace IDs from Okahu filtered by a fact.

        Uses:  GET /api/v1/workflows/<wf>/traces?duration_fact=<fact_name>&fact_ids=<fact_id>

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            fact_name: The fact to filter by (e.g. ``agentic_session``).
            fact_id: The fact value (e.g. a session ID).
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds.

        Returns:
            A list of trace ID strings.
        """
        base = OkahuSpanLoader._get_api_base(endpoint)
        headers = OkahuSpanLoader._get_headers(api_key)
        url = f"{base}/api/v1/workflows/{workflow_name}/traces"
        params = {
            "duration_fact": fact_name,
            "fact_ids": fact_id,
        }

        data = OkahuSpanLoader._do_get(
            url, headers, params=params, timeout=timeout,
            context_msg=f"traces for {fact_name}='{fact_id}' in workflow '{workflow_name}'"
        )

        trace_list = OkahuSpanLoader._unwrap_list(
            data, ("traces", "data", "results"),
            context_msg=f"traces for {fact_name}='{fact_id}'"
        )

        trace_ids = []
        for item in trace_list:
            if isinstance(item, dict) and "trace_id" in item:
                trace_ids.append(item["trace_id"])
            elif isinstance(item, str):
                trace_ids.append(item)

        logger.debug(
            "Found %d trace(s) for %s='%s' in workflow '%s'",
            len(trace_ids), fact_name, fact_id, workflow_name,
        )
        return trace_ids

    @staticmethod
    def get_spans(
        workflow_name: str,
        trace_id: str,
        filter_fact: Optional[str] = None,
        filter_fact_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> List[ReadableSpan]:
        """Fetch spans from Okahu for a given trace_id.

        Uses:  GET /api/v1/workflows/<wf>/traces/<trace_id>/spans
        Optionally appends ``?filter_fact=<fact>&filter_fact_id=<id>``
        to filter spans server-side (e.g. by session).

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            trace_id: The trace ID (hex string) to fetch spans for.
            filter_fact: Optional server-side span filter fact name.
            filter_fact_id: Optional server-side span filter fact value.
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds.

        Returns:
            A list of ReadableSpan instances.

        Raises:
            ValueError: If OKAHU_API_KEY is not configured.
            ConnectionError: If the request to Okahu fails.
        """
        # Strip 0x prefix if present
        trace_id = trace_id.replace("0x", "")

        base = OkahuSpanLoader._get_api_base(endpoint)
        headers = OkahuSpanLoader._get_headers(api_key)
        url = f"{base}/api/v1/workflows/{workflow_name}/traces/{trace_id}/spans"

        params = {}
        if filter_fact and filter_fact_id:
            params["filter_fact"] = filter_fact
            params["filter_fact_id"] = filter_fact_id

        span_data_list = OkahuSpanLoader._do_get(
            url, headers, params=params or None, timeout=timeout,
            context_msg=f"spans for trace_id '{trace_id}' in workflow '{workflow_name}'"
        )

        span_data_list = OkahuSpanLoader._unwrap_list(
            span_data_list, ("spans", "batch", "data", "results", "trace_spans"),
            context_msg=f"spans for trace_id '{trace_id}'"
        )

        span_list = []
        for item in span_data_list:
            span = JSONSpanLoader._from_dict(span_data=item)
            span_list.append(span)
        # verify that there's a span with span.attributes["span.type"] == "workflow" otherwise raise HttpError 404
        if not any(span.attributes.get("span.type") == "workflow" for span in span_list):
            raise requests.HTTPError(f"No workflow span found in trace '{trace_id}' - possible invalid trace ID or trace not fully ingested yet.")

        logger.debug("Loaded %d spans from Okahu for trace_id '%s'", len(span_list), trace_id)
        return span_list

    @staticmethod
    def load_by_session(
        workflow_name: str,
        session_id: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> List[ReadableSpan]:
        """Fetch all spans for every trace in a session.

        This is a convenience wrapper around ``load_by_scope()`` that uses
        the standard "agent_sessions" scope name.

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            session_id: The agent session ID.
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds.

        Returns:
            A flat list of ReadableSpan instances from all matching traces.

        Raises:
            ConnectionError: If no traces found or API call fails.
        """
        return OkahuSpanLoader.load_by_scope(
            workflow_name=workflow_name,
            scope_name=OkahuSpanLoader.AGENT_SESSIONS_SCOPE,
            scope_id=session_id,
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
        )

    @staticmethod
    def load_by_scope(
        workflow_name: str,
        scope_name: str,
        scope_id: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> List[ReadableSpan]:
        """Fetch all spans for every trace matching a custom scope.

        This is a generic method that works with any Okahu fact/scope.
        For example:
        - scope_name="agent_sessions", scope_id="session_123"
        - scope_name="test_id", scope_id="test_456"
        - scope_name="my_custom_scope", scope_id="custom_789"

        1. GET traces with ``duration_fact=<scope_name>&fact_ids=<scope_id>``
        2. For each trace, GET spans with ``filter_fact=<scope_name>&filter_fact_id=<scope_id>``
        3. Return ReadableSpan objects.

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            scope_name: The name of the scope/fact to filter by.
            scope_id: The scope/fact value (e.g., session ID, test ID, etc.).
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds.

        Returns:
            A flat list of ReadableSpan instances from all matching traces.

        Raises:
            ValueError: If scope_name or scope_id is empty.
            ConnectionError: If no traces found or API call fails.
        """
        # Validate inputs
        if not scope_name or not scope_name.strip():
            raise ValueError("scope_name cannot be empty")
        if not scope_id or not scope_id.strip():
            raise ValueError("scope_id cannot be empty")

        trace_ids = OkahuSpanLoader.get_trace_ids(
            workflow_name,
            fact_name=scope_name,
            fact_id=scope_id,
            endpoint=endpoint, api_key=api_key, timeout=timeout,
        )
        if not trace_ids:
            raise ConnectionError(
                f"No traces found for {scope_name}='{scope_id}' in workflow '{workflow_name}'"
            )

        all_spans: List[ReadableSpan] = []
        for tid in trace_ids:
            spans = OkahuSpanLoader.get_spans(
                workflow_name, tid,
                filter_fact=scope_name,
                filter_fact_id=scope_id,
                endpoint=endpoint, api_key=api_key, timeout=timeout,
            )
            all_spans.extend(spans)

        logger.debug(
            "Loaded %d total spans across %d trace(s) for %s='%s'",
            len(all_spans), len(trace_ids), scope_name, scope_id,
        )
        return all_spans
