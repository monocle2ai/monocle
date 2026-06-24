import os
import json
import datetime
import unittest
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult


def _make_span(name="test-span", trace_id=0xAABBCCDD, span_id=0x1111,
               parent_span_id=None):
    """Build a minimal mock ReadableSpan for use in tests."""
    span = MagicMock(spec=ReadableSpan)
    span.name = name
    span.context = MagicMock()
    span.context.trace_id = trace_id
    span.context.span_id = span_id
    span.parent = None
    if parent_span_id is not None:
        span.parent = MagicMock()
        span.parent.span_id = parent_span_id
    span.start_time = 1_000_000_000   # 1 s past epoch, in nanoseconds
    span.end_time   = 2_000_000_000
    span.attributes = MagicMock()
    span.attributes.get.return_value = "0.8.0"   # non-empty → not skipped
    span.to_json.return_value = json.dumps({
        "name": name,
        "status": {"status_code": "OK"},
        "attributes": {"monocle_apptrace.version": "0.8.0"},
        "events": [],
    })
    return span


class TestPostgresRegistry(unittest.TestCase):
    def test_postgres_in_registry(self):
        from monocle_apptrace.exporters.monocle_exporters import monocle_exporters
        self.assertIn("postgres", monocle_exporters)
        entry = monocle_exporters["postgres"]
        self.assertEqual(entry["module"], "monocle_apptrace.exporters.postgres.postgres_exporter")
        self.assertEqual(entry["class"], "PostgresSpanExporter")


class TestPostgresInit(unittest.TestCase):
    @patch("psycopg2.connect")
    def test_reads_connection_url_from_env(self, mock_connect):
        os.environ["MONOCLE_POSTGRES_CONNECTION_URL"] = "postgresql://user:pass@localhost/db"
        from monocle_apptrace.exporters.postgres.postgres_exporter import PostgresSpanExporter
        exporter = PostgresSpanExporter()
        mock_connect.assert_called_once_with("postgresql://user:pass@localhost/db")
        self.assertIsNotNone(exporter.connection)
        del os.environ["MONOCLE_POSTGRES_CONNECTION_URL"]

    @patch("psycopg2.connect")
    def test_raises_when_url_missing(self, mock_connect):
        os.environ.pop("MONOCLE_POSTGRES_CONNECTION_URL", None)
        from monocle_apptrace.exporters.postgres.postgres_exporter import PostgresSpanExporter
        with self.assertRaises(ValueError):
            PostgresSpanExporter()
        mock_connect.assert_not_called()
