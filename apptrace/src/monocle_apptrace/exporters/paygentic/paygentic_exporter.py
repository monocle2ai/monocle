"""Paygentic span exporter -- sends CloudEvents to the Paygentic API.

Derives the CloudEvent type from monocle span attributes (entity.1.type)
so each provider/span kind gets its own type automatically.

Extends SpanExporterBase for consistency with other monocle exporters
and reuses its retry_with_backoff decorator for transient-error resilience.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

import requests
from monocle_apptrace.exporters.base_exporter import SpanExporterBase
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor
from monocle_apptrace.exporters.span_filter import SpanFilter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

logger = logging.getLogger(__name__)

PAYGENTIC_PROD_ENDPOINT = "https://api.paygentic.io/v0/events"
PAYGENTIC_SANDBOX_ENDPOINT = "https://api.sandbox.paygentic.io/v0/events"
SUCCESS_STATUS_CODES = {200, 202}

# Monocle sets scopes as span attributes with "scope." prefix
# (baggage key "monocle.scope.X" -> span attr "scope.X")
SCOPE_SUBSCRIPTION_ID = "scope.subscriptionId"
SCOPE_CUSTOMER_ID = "scope.customerId"

# How long (seconds) to suppress a rejected event type before retrying
REJECTED_TYPE_COOLDOWN = 3600  # 1 hour


class PaygenticSpanExporter(SpanExporterBase):
    """OpenTelemetry SpanExporter that sends CloudEvents to the Paygentic API.

    The CloudEvent type is derived from the span's entity.1.type attribute
    (e.g. "inference.vertexai" -> "ai.inference.vertexai").
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: Optional[int] = None,
        task_processor: Optional[ExportTaskProcessor] = None,
        source: Optional[str] = None,
        namespace: Optional[str] = None,
        sandbox: Optional[bool] = None,
        span_filter_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the exporter.

        Args:
            span_filter_config: SpanFilter configuration dict.  Defaults to
                inference spans with only ``entity.1.type``, ``scope.*``
                attributes and ``metadata`` events.  Custom configs must
                keep ``scope.*`` in ``fields_to_include.attributes`` or
                ``_resolve_subject`` will fail to find a subject for every span.
        """
        super().__init__()
        api_key = os.environ.get("PAYGENTIC_API_KEY")
        if not api_key:
            raise ValueError("PAYGENTIC_API_KEY not set.")

        is_sandbox = (
            sandbox
            if sandbox is not None
            else os.environ.get("PAYGENTIC_SANDBOX", "").lower() == "true"
        )
        self._endpoint = (
            endpoint
            or os.environ.get("PAYGENTIC_EVENT_ENDPOINT")
            or (PAYGENTIC_SANDBOX_ENDPOINT if is_sandbox else PAYGENTIC_PROD_ENDPOINT)
        )
        self._timeout = timeout or int(os.environ.get("PAYGENTIC_TIMEOUT", "15"))
        self._source = source or os.environ.get("PAYGENTIC_SOURCE", "monocle")
        self._namespace = namespace or os.environ.get("PAYGENTIC_NAMESPACE")
        self._closed = False
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # event types rejected by the API (type -> timestamp of rejection)
        self._rejected_types: Dict[str, float] = {}

        self.task_processor = task_processor
        if task_processor is not None:
            task_processor.start()

        self._span_filter = SpanFilter(
            span_filter_config or {
                "span_types_to_include": ["inference", "inference.*"],
                "fields_to_include": {
                    "attributes": ["entity.1.type", "scope.*"],
                    "events": [{"name": "metadata"}],
                },
            }
        )

        logger.info("Initializing with endpoint: %s", self._endpoint)

    @staticmethod
    def _resolve_subject(
        attributes: Dict[str, Any], span_name: str, trace_id: str,
    ) -> Optional[str]:
        """Resolve the event subject from monocle scopes.

        Priority: subscriptionId scope > customerId scope.
        Returns None and logs an error when no subject can be determined.
        """
        subscription_id = attributes.get(SCOPE_SUBSCRIPTION_ID)
        if subscription_id:
            return str(subscription_id)

        customer_id = attributes.get(SCOPE_CUSTOMER_ID)
        if customer_id:
            return str(customer_id)

        logger.error(
            "No subscriptionId or customerId scope set on span %s (trace %s). "
            "Set a monocle scope for 'subscriptionId' or 'customerId' before "
            "the traced call.",
            span_name,
            trace_id,
        )
        return None

    @staticmethod
    def _derive_type(attributes: Dict[str, Any]) -> str:
        """Derive CloudEvent type from the entity.1.type attribute.

        e.g. "inference.vertexai" -> "ai.inference.vertexai"
        """
        entity_type = attributes.get("entity.1.type", "unknown")
        return f"ai.{entity_type}"

    def _is_type_rejected(self, event_type: str) -> bool:
        """Check if an event type is temporarily rejected (cooldown not expired)."""
        rejected_at = self._rejected_types.get(event_type)
        if rejected_at is None:
            return False
        if time.monotonic() - rejected_at > REJECTED_TYPE_COOLDOWN:
            self._rejected_types.pop(event_type, None)
            logger.info("Cooldown expired for event type %r, will retry", event_type)
            return False
        return True

    def _reject_type(self, event_type: str) -> None:
        """Mark an event type as rejected by the API."""
        if event_type not in self._rejected_types:
            logger.warning(
                "Event type %r rejected by API, suppressing for %ds",
                event_type,
                REJECTED_TYPE_COOLDOWN,
            )
        self._rejected_types[event_type] = time.monotonic()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._closed:
            logger.warning("Exporter already shutdown, ignoring batch")
            return SpanExportResult.FAILURE

        if not spans:
            return SpanExportResult.SUCCESS

        events = []
        for span in spans:
            if self.skip_export(span):
                continue

            filtered = self._span_filter.filter(span)
            if filtered is None:
                continue

            attrs = filtered.get("attributes", {})
            ctx = filtered.get("context", {})
            trace_id = ctx.get("trace_id", "")

            # Events are already filtered to metadata-only by SpanFilter
            metadata_attrs: Dict[str, Any] = {}
            for ev in filtered.get("events", []):
                metadata_attrs.update(ev.get("attributes", {}))
            if not metadata_attrs:
                continue

            event_type = self._derive_type(attrs)
            if self._is_type_rejected(event_type):
                continue

            subject = self._resolve_subject(attrs, filtered.get("name", ""), trace_id)
            if subject is None:
                continue

            event: Dict[str, Any] = {
                "type": event_type,
                "source": self._source,
                "subject": subject,
                "idempotencyKey": f"{trace_id}_{ctx.get('span_id', '')}",
                "data": {
                    "span_id": ctx.get("span_id", ""),
                    "trace_id": trace_id,
                    "name": filtered.get("name"),
                    **metadata_attrs,
                },
            }
            if self._namespace:
                event["namespace"] = self._namespace
            events.append(event)

        if not events:
            return SpanExportResult.SUCCESS

        is_root_span = any(not span.parent for span in spans)

        if self.task_processor is not None and callable(self.task_processor.queue_task):
            self.task_processor.queue_task(
                self._send_events,
                kwargs={"events": events},
                is_root_span=is_root_span,
            )
            return SpanExportResult.SUCCESS

        return self._send_events(events)

    def _send_events(self, events: List[Dict[str, Any]]) -> SpanExportResult:
        logger.info("Sending %d event(s) to %s", len(events), self._endpoint)

        failed = 0
        with ThreadPoolExecutor(max_workers=min(len(events), 10)) as pool:
            futures = {
                pool.submit(self._post_single_event, event): event
                for event in events
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Export request failed: %s", exc)
                    failed += 1

        if failed > 0:
            logger.error("%d/%d event(s) failed", failed, len(events))
            return SpanExportResult.FAILURE

        logger.info("Successfully exported %d event(s)", len(events))
        return SpanExportResult.SUCCESS

    @SpanExporterBase.retry_with_backoff(
        exceptions=(requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    )
    def _post_single_event(self, event: Dict[str, Any]) -> None:
        resp = requests.post(
            self._endpoint, json=event, headers=self._headers,
            timeout=self._timeout,
        )
        if resp.status_code == 429:
            raise requests.exceptions.ConnectionError("Rate limited")
        if resp.status_code >= 500:
            raise requests.exceptions.ConnectionError(
                f"Server error {resp.status_code}"
            )
        if resp.status_code == 422:
            try:
                body = resp.json()
            except Exception:
                body = {}
            if body.get("error") == "invalid_event_type":
                self._reject_type(event["type"])
                return
        if resp.status_code not in SUCCESS_STATUS_CODES:
            logger.error(
                "Export failed - Status: %d, Response: %s",
                resp.status_code,
                resp.text,
            )
            raise ValueError(f"Client error {resp.status_code}")

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info("Shut down")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
