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


import psycopg2

class TestInsert(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_POSTGRES_CONNECTION_URL"] = "postgresql://u:p@h/db"
        with patch("psycopg2.connect"):
            from importlib import reload
            import monocle_apptrace.exporters.postgres.postgres_exporter as m
            reload(m)
            self.exporter = m.PostgresSpanExporter()
        # Set up a reusable mock cursor as a context manager
        self.mock_cursor = MagicMock()
        self.exporter.connection.cursor.return_value.__enter__ = lambda s: self.mock_cursor
        self.exporter.connection.cursor.return_value.__exit__ = MagicMock(return_value=False)

    def tearDown(self):
        os.environ.pop("MONOCLE_POSTGRES_CONNECTION_URL", None)

    def test_insert_trace_calls_execute_values(self):
        self.exporter.trace_spans[0xAABB] = ([_make_span()], datetime.datetime.now(), True)
        with patch("psycopg2.extras.execute_values") as mock_ev:
            self.exporter._insert_trace(0xAABB)
        mock_ev.assert_called_once()
        args = mock_ev.call_args[0]
        self.assertEqual(args[0], self.mock_cursor)
        self.assertIn("INSERT INTO traces", args[1])
        self.assertEqual(len(args[2]), 1)

    def test_insert_trace_removes_from_buffer(self):
        self.exporter.trace_spans[0xAABB] = ([_make_span()], datetime.datetime.now(), True)
        with patch("psycopg2.extras.execute_values"):
            self.exporter._insert_trace(0xAABB)
        self.assertNotIn(0xAABB, self.exporter.trace_spans)

    def test_bad_span_skipped_good_span_inserted(self):
        good_span = _make_span(span_id=0x1111)
        bad_span  = _make_span(span_id=0x2222)
        bad_span.to_json.side_effect = Exception("serialization error")
        self.exporter.trace_spans[0xAABB] = ([good_span, bad_span],
                                              datetime.datetime.now(), True)
        with patch("psycopg2.extras.execute_values") as mock_ev:
            self.exporter._insert_trace(0xAABB)
        rows = mock_ev.call_args[0][2]
        self.assertEqual(len(rows), 1)

    @patch("psycopg2.connect")
    def test_reconnects_and_retries_on_operational_error(self, mock_connect):
        self.exporter.trace_spans[0xAABB] = ([_make_span()], datetime.datetime.now(), True)
        call_count = {"n": 0}

        def do_insert_side_effect(rows):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise psycopg2.OperationalError("server closed connection")

        with patch.object(self.exporter, "_do_insert", side_effect=do_insert_side_effect):
            self.exporter._insert_trace(0xAABB)

        mock_connect.assert_called()
        self.assertEqual(call_count["n"], 2)
        self.assertNotIn(0xAABB, self.exporter.trace_spans)
