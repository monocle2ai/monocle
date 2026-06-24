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


class TestBuildRow(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_POSTGRES_CONNECTION_URL"] = "postgresql://u:p@h/db"
        with patch("psycopg2.connect"):
            from importlib import reload
            import monocle_apptrace.exporters.postgres.postgres_exporter as m
            reload(m)
            self.exporter = m.PostgresSpanExporter()

    def tearDown(self):
        os.environ.pop("MONOCLE_POSTGRES_CONNECTION_URL", None)

    def test_span_id_has_0x_prefix(self):
        row = self.exporter._build_row(_make_span(span_id=0xABCDEF0123456789))
        self.assertEqual(row[4], "0xabcdef0123456789")

    def test_trace_id_has_0x_prefix(self):
        row = self.exporter._build_row(_make_span(trace_id=0xAABBCCDD11223344AABBCCDD11223344))
        self.assertTrue(row[5].startswith("0x"))

    def test_parent_id_is_none_for_root_span(self):
        row = self.exporter._build_row(_make_span())  # no parent
        self.assertIsNone(row[6])

    def test_parent_id_has_0x_prefix_for_child_span(self):
        row = self.exporter._build_row(_make_span(parent_span_id=0x1111222233334444))
        self.assertEqual(row[6], "0x1111222233334444")

    def test_timestamps_are_timezone_aware_datetimes(self):
        row = self.exporter._build_row(_make_span())
        self.assertIsInstance(row[1], datetime.datetime)
        self.assertIsInstance(row[2], datetime.datetime)
        self.assertIsNotNone(row[1].tzinfo)

    def test_metadata_is_none(self):
        row = self.exporter._build_row(_make_span())
        self.assertIsNone(row[9])


class TestSpanBuffering(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_POSTGRES_CONNECTION_URL"] = "postgresql://u:p@h/db"
        with patch("psycopg2.connect"):
            from importlib import reload
            import monocle_apptrace.exporters.postgres.postgres_exporter as m
            reload(m)
            self.exporter = m.PostgresSpanExporter()

    def tearDown(self):
        os.environ.pop("MONOCLE_POSTGRES_CONNECTION_URL", None)

    def test_spans_accumulated_across_calls(self):
        span1 = _make_span(span_id=0x1111)
        span2 = _make_span(span_id=0x2222)
        self.exporter._add_spans_to_trace(0xAABB, [span1], has_root=False)
        self.exporter._add_spans_to_trace(0xAABB, [span2], has_root=True)
        buffered_spans, _, has_root = self.exporter.trace_spans[0xAABB]
        self.assertEqual(len(buffered_spans), 2)
        self.assertTrue(has_root)

    def test_expired_trace_flushed(self):
        old_time = datetime.datetime.now() - datetime.timedelta(seconds=61)
        self.exporter.trace_spans[0xAABB] = ([_make_span()], old_time, False)
        with patch.object(self.exporter, "_insert_trace") as mock_insert:
            self.exporter._cleanup_expired_traces()
        mock_insert.assert_called_once_with(0xAABB)

    def test_non_expired_trace_not_flushed(self):
        self.exporter.trace_spans[0xAABB] = ([_make_span()], datetime.datetime.now(), False)
        with patch.object(self.exporter, "_insert_trace") as mock_insert:
            self.exporter._cleanup_expired_traces()
        mock_insert.assert_not_called()
