import os
import json
import logging
import datetime
from typing import Sequence

import clickhouse_connect
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from monocle_apptrace.exporters.base_exporter import (
    SpanExporterBase,
    format_span_id_without_0x,
    format_trace_id_without_0x,
    serialize_span,
)

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS traces (
        name        String,
        start_time  DateTime64(9, 'UTC'),
        end_time    DateTime64(9, 'UTC'),
        status      String,
        span_id     String,
        trace_id    String,
        parent_id   Nullable(String),
        attributes  String,
        events      String,
        metadata    Nullable(String)
    ) ENGINE = MergeTree() ORDER BY (trace_id, span_id)
"""

INSERT_COLUMNS = [
    "name", "start_time", "end_time", "status", "span_id", "trace_id",
    "parent_id", "attributes", "events", "metadata",
]

# ClickHouse error code raised when the user lacks the required privilege.
ACCESS_DENIED_CODE = "497"


class ClickHouseSpanExporter(SpanExporterBase):

    def __init__(self) -> None:
        super().__init__()
        self.connection_url = os.environ.get("MONOCLE_CLICKHOUSE_CONNECTION_URL")
        if not self.connection_url:
            raise ValueError("MONOCLE_CLICKHOUSE_CONNECTION_URL environment variable is required")
        self.client = clickhouse_connect.get_client(dsn=self.connection_url)
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            self.client.command(CREATE_TABLE_SQL)
        except DatabaseError as e:
            if ACCESS_DENIED_CODE in str(e) or "ACCESS_DENIED" in str(e):
                raise PermissionError(
                    "ClickHouseSpanExporter could not create the 'traces' table — "
                    "the database user lacks CREATE TABLE permission. "
                    "Pre-create the table manually or grant the required privilege."
                ) from e
            raise

    def _build_row(self, span: ReadableSpan) -> list:
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
        return [
            span.name,
            start_time,
            end_time,
            json.dumps(serialized.get("status")),
            span_id,
            trace_id,
            parent_id,
            json.dumps(serialized.get("attributes")),
            json.dumps(serialized.get("events")),
            None,   # metadata — reserved for future use
        ]

    def _reconnect(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        self.client = clickhouse_connect.get_client(dsn=self.connection_url)

    def _do_insert(self, rows: list) -> None:
        self.client.insert("traces", rows, column_names=INSERT_COLUMNS)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
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
                except OperationalError as e:
                    logger.warning("DB connection error, attempting reconnect: %s", e)
                    self._reconnect()
                    self._do_insert(rows)

            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error("Error exporting spans to ClickHouse: %s", e)
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        logger.info("ClickHouseSpanExporter has been shut down.")
