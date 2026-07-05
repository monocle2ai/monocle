import os
import logging
import datetime
from typing import Dict, List, Sequence, Tuple

import psycopg2
import psycopg2.errors
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

POSTGRESEXPORTER_HANDLE_TIMEOUT_SECONDS = int(
    os.environ.get("MONOCLE_POSTGRESEXPORTER_HANDLE_TIMEOUT_SECONDS", 60)
)

CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS traces (
        id          BIGSERIAL PRIMARY KEY,
        name        TEXT NOT NULL,
        start_time  TIMESTAMPTZ,
        end_time    TIMESTAMPTZ,
        status      JSONB,
        span_id     TEXT,
        trace_id    TEXT,
        parent_id   TEXT,
        attributes  JSONB,
        events      JSONB,
        metadata    JSONB
    )
"""

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
        self._ensure_table()
        self.trace_spans: Dict[int, Tuple[List[ReadableSpan], datetime.datetime, bool]] = {}

    def _ensure_table(self) -> None:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)
            self.connection.commit()
        except psycopg2.errors.InsufficientPrivilege as e:
            self.connection.rollback()
            raise PermissionError(
                "PostgresSpanExporter could not create the 'traces' table — "
                "the database user lacks CREATE TABLE permission. "
                "Pre-create the table manually or grant the required privilege."
            ) from e

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
            self.trace_spans[trace_id] = (spans.copy(), datetime.datetime.now(tz=datetime.timezone.utc), has_root)

    def _cleanup_expired_traces(self) -> None:
        current_time = datetime.datetime.now(tz=datetime.timezone.utc)
        expired = [
            trace_id
            for trace_id, (_, creation_time, _) in self.trace_spans.items()
            if (current_time - creation_time).total_seconds() > POSTGRESEXPORTER_HANDLE_TIMEOUT_SECONDS
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
                logger.warning("Error serializing span %s: %s", span.context.span_id, e)
        if rows:
            try:
                self._do_insert(rows)
            except psycopg2.OperationalError as e:
                logger.warning("DB connection error, attempting reconnect: %s", e)
                self._reconnect()
                self._do_insert(rows)
        del self.trace_spans[trace_id]

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            self._cleanup_expired_traces()
            spans_by_trace: Dict[int, List[ReadableSpan]] = {}
            root_span_traces: set = set()

            for span in spans:
                if self.skip_export(span):
                    continue
                trace_id = span.context.trace_id
                spans_by_trace.setdefault(trace_id, []).append(span)
                if not span.parent:
                    root_span_traces.add(trace_id)

            for trace_id, trace_spans in spans_by_trace.items():
                self._add_spans_to_trace(trace_id, trace_spans,
                                          trace_id in root_span_traces)

            for trace_id in root_span_traces:
                self._insert_trace(trace_id)

            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error("Error exporting spans to Postgres: %s", e)
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        for trace_id in list(self.trace_spans.keys()):
            try:
                self._insert_trace(trace_id)
            except Exception as e:
                logger.error("Error flushing trace %s: %s", format_trace_id_without_0x(trace_id), e)
        return True

    def shutdown(self) -> None:
        self.force_flush()
        try:
            self.connection.close()
        except Exception:
            pass
        logger.info("PostgresSpanExporter has been shut down.")
