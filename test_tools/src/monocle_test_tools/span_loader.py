import datetime
import json
import logging
import os
import glob
from typing import Any, Dict, List, Optional
import requests
from opentelemetry.sdk.trace import ReadableSpan, Status, StatusCode, Event, Resource
from opentelemetry import trace as trace_api
from monocle_apptrace.exporters.file_exporter import DEFAULT_TRACE_FOLDER

logger = logging.getLogger(__name__)


class JSONSpanLoader:
    """Utility class to load spans from JSON trace files."""

    @staticmethod
    def from_json(json_file_path: str) -> List[ReadableSpan]:
        """Load spans from a JSON file.

        Args:
            json_file_path: Absolute path to the JSON file containing span data.

        Returns:
            A list of ReadableSpan instances.
        """
        span_list = []
        with open(json_file_path, 'r') as f:
            span_data = json.load(f)
            for item in span_data:
                span = JSONSpanLoader._from_dict(span_data=item)
                span_list.append(span)
        return span_list

    @staticmethod
    def find_trace_file(trace_id: str, trace_dir: Optional[str] = None) -> Optional[str]:
        """Find a trace file matching the given trace_id.

        The file naming convention is:
            monocle_trace_{service_name}_{trace_id}_{timestamp}.json

        Args:
            trace_id: The trace ID to search for (hex string without 0x prefix).
            trace_dir: Directory to search in. Defaults to .monocle/test_traces.

        Returns:
            The file path if found, None otherwise.
        """
        if trace_dir is None:
            trace_dir = os.path.join(".", DEFAULT_TRACE_FOLDER, "test_traces")

        # Strip 0x prefix if present
        trace_id = trace_id.replace("0x", "")

        pattern = os.path.join(trace_dir, f"*_{trace_id}_*.json")
        matches = glob.glob(pattern)
        if matches:
            # Return the most recent file if multiple matches
            return max(matches, key=os.path.getmtime)
        return None

    @staticmethod
    def _from_dict(span_data: Dict[str, Any]) -> ReadableSpan:
        """Create a ReadableSpan instance from a dictionary.

        Handles both the local file export format and the Okahu API format:
        - File format: name, context.trace_id, context.span_id, parent_id, kind, resource, links
        - Okahu format: span_name, trace_id, span_id (top-level), no parent_id/kind/resource/links
        """
        # Resolve span name: "name" (file format) or "span_name" (Okahu format)
        span_name = span_data.get("name") or span_data.get("span_name", "unknown")

        # Parse context — file format uses nested "context" dict, Okahu uses top-level fields
        context = None
        if span_data.get("context"):
            context = JSONSpanLoader._parse_context(span_data["context"])
        elif span_data.get("trace_id") and span_data.get("span_id"):
            trace_id_str = str(span_data["trace_id"]).replace("0x", "")
            span_id_str = str(span_data["span_id"]).replace("0x", "")
            context = trace_api.SpanContext(
                trace_id=int(trace_id_str, 16),
                span_id=int(span_id_str, 16),
                is_remote=False,
                trace_flags=trace_api.TraceFlags.DEFAULT,
                trace_state=None,
            )

        # Parse parent context
        parent = None
        parent_id_str = span_data.get("parent_id") or span_data.get("parent_span_id")
        if parent_id_str:
            parent_span_id = int(str(parent_id_str).replace("0x", ""), 16)
            if context:
                parent = trace_api.SpanContext(
                    trace_id=context.trace_id,
                    span_id=parent_span_id,
                    is_remote=True,
                    trace_flags=trace_api.TraceFlags.DEFAULT,
                    trace_state=None
                )

        # Parse timestamps
        start_time = None
        if span_data.get("start_time"):
            start_time = JSONSpanLoader._iso_str_to_ns(span_data["start_time"])

        end_time = None
        if span_data.get("end_time"):
            end_time = JSONSpanLoader._iso_str_to_ns(span_data["end_time"])

        # Parse status
        status = Status(StatusCode.UNSET)
        if span_data.get("status"):
            status_data = span_data["status"]
            if isinstance(status_data, str):
                # Handle plain string status e.g. "OK", "ok", "ERROR", "UNSET"
                status_code = getattr(StatusCode, status_data.upper(), StatusCode.UNSET)
                status = Status(status_code)
            elif isinstance(status_data, dict):
                # Handle dict status e.g. {"status_code": "OK"} or {"code": "ok"}
                code_str = (status_data.get("status_code") or status_data.get("code") or "UNSET").upper()
                status_code = getattr(StatusCode, code_str, StatusCode.UNSET)
                description = status_data.get("description") or status_data.get("message")
                status = Status(status_code, description)

        # Parse kind
        kind = trace_api.SpanKind.INTERNAL
        if span_data.get("kind"):
            kind_str = span_data["kind"].replace("SpanKind.", "")
            kind = getattr(trace_api.SpanKind, kind_str, trace_api.SpanKind.INTERNAL)

        # Parse events
        events = []
        if span_data.get("events"):
            for event_data in span_data["events"]:
                timestamp = JSONSpanLoader._iso_str_to_ns(event_data["timestamp"])
                event = Event(
                    name=event_data["name"],
                    attributes=event_data.get("attributes"),
                    timestamp=timestamp
                )
                events.append(event)

        # Parse links
        links = []
        if span_data.get("links"):
            for link_data in span_data["links"]:
                link_context = JSONSpanLoader._parse_context(link_data["context"])
                link = trace_api.Link(
                    context=link_context,
                    attributes=link_data.get("attributes")
                )
                links.append(link)

        # Parse resource
        resource = None
        if span_data.get("resource"):
            resource = Resource.create(span_data["resource"].get("attributes", {}))

        # Normalise attributes – flatten structured "entity" list from Okahu
        # into the flat "entity.N.key" format used by the validator.
        attributes = span_data.get("attributes") or {}
        attributes = JSONSpanLoader._normalize_attributes(attributes)

        return ReadableSpan(
            name=span_name,
            context=context,
            parent=parent,
            resource=resource,
            attributes=attributes,
            events=events,
            links=links,
            kind=kind,
            status=status,
            start_time=start_time,
            end_time=end_time
        )

    @staticmethod
    def _parse_context(context_data: Dict[str, str]) -> trace_api.SpanContext:
        """Parse span context from dictionary."""
        trace_id = int(context_data["trace_id"], 16)
        span_id = int(context_data["span_id"], 16)
        return trace_api.SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=trace_api.TraceFlags.DEFAULT,
            trace_state=None
        )

    @staticmethod
    def _iso_str_to_ns(iso_str: str) -> int:
        """Convert ISO format timestamp string to nanoseconds."""
        dt = datetime.datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return int(dt.timestamp() * 1e9)

    @staticmethod
    def _normalize_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise attributes from Okahu structured format to flat format.

        Okahu returns entities as:
            {"entity": [{"name": "foo", "type": "tool.adk"}, ...], "entity.count": 2}
        The validator expects:
            {"entity.1.name": "foo", "entity.1.type": "tool.adk", "entity.count": 2}

        This method converts the structured list into the flattened format while
        preserving all other attributes unchanged.
        """
        if not isinstance(attributes, dict):
            return attributes

        entity_list = attributes.get("entity")
        if not isinstance(entity_list, list):
            return attributes

        # Check if the list contains dicts (Okahu structured format)
        if not entity_list or not isinstance(entity_list[0], dict):
            return attributes

        normalised = {k: v for k, v in attributes.items() if k != "entity"}
        for idx, entity in enumerate(entity_list, start=1):
            if isinstance(entity, dict):
                for key, value in entity.items():
                    normalised[f"entity.{idx}.{key}"] = value
        if "entity.count" not in normalised:
            normalised["entity.count"] = len(entity_list)
        return normalised


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

    @staticmethod
    def _get_api_base(endpoint: Optional[str] = None) -> str:
        """Return the Okahu API base URL (no trailing slash)."""
        return (endpoint or os.environ.get("OKAHU_API_ENDPOINT", '')).rstrip("/")

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
            raise ConnectionError(
                f"Okahu request failed ({context_msg}). "
                f"HTTP {response.status_code}: {response.text}"
            ) from exc
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
