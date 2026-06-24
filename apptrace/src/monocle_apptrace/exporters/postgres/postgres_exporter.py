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

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        raise NotImplementedError

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        raise NotImplementedError
