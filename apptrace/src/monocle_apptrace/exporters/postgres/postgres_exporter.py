import os
import logging
import datetime
from typing import Dict, List, Sequence, Tuple

import psycopg2
import psycopg2.extras

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from monocle_apptrace.exporters.base_exporter import (
    SpanExporterBase,
    format_span_id_without_0x,
    format_trace_id_without_0x,
    serialize_span,
)

logger = logging.getLogger(__name__)

HANDLE_TIMEOUT_SECONDS = 60

INSERT_SQL = """
    INSERT INTO traces
        (name, start_time, end_time, status, span_id, trace_id,
         parent_id, attributes, events, metadata)
    VALUES %s
"""


class PostgresSpanExporter(SpanExporterBase):

    def __init__(self) -> None:
        super().__init__()
        self.connection_url = os.environ.get("MONOCLE_POSTGRES_CONNECTION_URL")
        if not self.connection_url:
            raise ValueError("MONOCLE_POSTGRES_CONNECTION_URL environment variable is required")
        self.connection = psycopg2.connect(self.connection_url)
        self.trace_spans: Dict[int, Tuple[List[ReadableSpan], datetime.datetime, bool]] = {}

    def _build_row(self, span: ReadableSpan) -> tuple:
        serialized = serialize_span(span)
        start_time = datetime.datetime.fromtimestamp(
            span.start_time / 1e9, tz=datetime.timezone.utc
        )
        end_time = datetime.datetime.fromtimestamp(
            span.end_time / 1e9, tz=datetime.timezone.utc
        )
        span_id  = "0x" + format_span_id_without_0x(span.context.span_id)
        trace_id = "0x" + format_trace_id_without_0x(span.context.trace_id)
        parent_id = (
            "0x" + format_span_id_without_0x(span.parent.span_id)
            if span.parent else None
        )
        return (
            span.name,
            start_time,
            end_time,
            psycopg2.extras.Json(serialized.get("status")),
            span_id,
            trace_id,
            parent_id,
            psycopg2.extras.Json(serialized.get("attributes")),
            psycopg2.extras.Json(serialized.get("events")),
            None,   # metadata — reserved for future use
        )

    def _add_spans_to_trace(self, trace_id: int, spans: List[ReadableSpan],
                             has_root: bool = False) -> None:
        if trace_id in self.trace_spans:
            existing_spans, creation_time, existing_root = self.trace_spans[trace_id]
            existing_spans.extend(spans)
            self.trace_spans[trace_id] = (existing_spans, creation_time,
                                           has_root or existing_root)
        else:
            self.trace_spans[trace_id] = (spans.copy(), datetime.datetime.now(), has_root)

    def _cleanup_expired_traces(self) -> None:
        current_time = datetime.datetime.now()
        expired = [
            trace_id
            for trace_id, (_, creation_time, _) in self.trace_spans.items()
            if (current_time - creation_time).total_seconds() > HANDLE_TIMEOUT_SECONDS
        ]
        for trace_id in expired:
            self._insert_trace(trace_id)

    def _reconnect(self) -> None:
        try:
            self.connection.close()
        except Exception:
            pass
        self.connection = psycopg2.connect(self.connection_url)

    def _do_insert(self, rows: list) -> None:
        with self.connection.cursor() as cursor:
            psycopg2.extras.execute_values(cursor, INSERT_SQL, rows)
        self.connection.commit()

    def _insert_trace(self, trace_id: int) -> None:
        if trace_id not in self.trace_spans:
            return
        spans, _, _ = self.trace_spans[trace_id]
        rows = []
        for span in spans:
            if self.skip_export(span):
                continue
            try:
                rows.append(self._build_row(span))
            except Exception as e:
                logger.warning(f"Error serializing span {span.context.span_id}: {e}")
        if rows:
            try:
                self._do_insert(rows)
            except psycopg2.OperationalError as e:
                logger.warning(f"DB connection error, attempting reconnect: {e}")
                self._reconnect()
                self._do_insert(rows)
        del self.trace_spans[trace_id]

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        raise NotImplementedError

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        raise NotImplementedError
