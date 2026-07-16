# pylint: disable=protected-access
import os
import json
import datetime
import unittest
from importlib import reload
from unittest.mock import MagicMock, patch
import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

# psycopg2 is an optional dependency (installed via the `postgres` extra).
psycopg2 = pytest.importorskip("psycopg2")
import psycopg2.errors
import monocle_apptrace.exporters.postgres.postgres_exporter as pg_mod
from monocle_apptrace.exporters.postgres.postgres_exporter import PostgresSpanExporter
from monocle_apptrace.exporters.monocle_exporters import monocle_exporters


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
        self.assertIn("postgres", monocle_exporters)
        entry = monocle_exporters["postgres"]
        self.assertEqual(entry["module"], "monocle_apptrace.exporters.postgres.postgres_exporter")
        self.assertEqual(entry["class"], "PostgresSpanExporter")


class TestPostgresInit(unittest.TestCase):
    @patch("psycopg2.connect")
    def test_reads_connection_url_from_env(self, mock_connect):
        os.environ["MONOCLE_POSTGRES_CONNECTION_URL"] = "postgresql://user:pass@localhost/db"
        exporter = PostgresSpanExporter()
        mock_connect.assert_called_once_with("postgresql://user:pass@localhost/db")
        self.assertIsNotNone(exporter.connection)
        del os.environ["MONOCLE_POSTGRES_CONNECTION_URL"]

    @patch("psycopg2.connect")
    def test_raises_when_url_missing(self, mock_connect):
        os.environ.pop("MONOCLE_POSTGRES_CONNECTION_URL", None)
        with self.assertRaises(ValueError):
            PostgresSpanExporter()
        mock_connect.assert_not_called()


class TestBuildRow(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_POSTGRES_CONNECTION_URL"] = "postgresql://u:p@h/db"
        with patch("psycopg2.connect"):
            reload(pg_mod)
            self.exporter = pg_mod.PostgresSpanExporter()

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


class TestExport(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_POSTGRES_CONNECTION_URL"] = "postgresql://u:p@h/db"
        with patch("psycopg2.connect"):
            reload(pg_mod)
            self.exporter = pg_mod.PostgresSpanExporter()
        # Set up a reusable mock cursor as a context manager
        self.mock_cursor = MagicMock()
        self.exporter.connection.cursor.return_value.__enter__ = lambda s: self.mock_cursor
        self.exporter.connection.cursor.return_value.__exit__ = MagicMock(return_value=False)

    def tearDown(self):
        os.environ.pop("MONOCLE_POSTGRES_CONNECTION_URL", None)

    def test_export_calls_execute_values_once_per_batch(self):
        child = _make_span(span_id=0x2222, parent_span_id=0x1111)
        root = _make_span(span_id=0x1111)

        with patch("psycopg2.extras.execute_values") as mock_ev:
            result = self.exporter.export([child, root])

        self.assertEqual(result, SpanExportResult.SUCCESS)
        mock_ev.assert_called_once()
        args = mock_ev.call_args[0]
        self.assertEqual(args[0], self.mock_cursor)
        self.assertIn("INSERT INTO traces", args[1])
        self.assertEqual(len(args[2]), 2)
        self.exporter.connection.commit.assert_called()

    def test_export_does_not_insert_when_no_rows(self):
        span = _make_span()
        span.attributes.get.return_value = None  # triggers skip_export

        with patch("psycopg2.extras.execute_values") as mock_ev:
            result = self.exporter.export([span])

        self.assertEqual(result, SpanExportResult.SUCCESS)
        mock_ev.assert_not_called()

    def test_export_returns_failure_on_exception(self):
        with patch.object(self.exporter, "skip_export", side_effect=RuntimeError("boom")):
            result = self.exporter.export([_make_span()])
        self.assertEqual(result, SpanExportResult.FAILURE)

    def test_bad_span_skipped_good_span_inserted(self):
        good_span = _make_span(span_id=0x1111)
        bad_span = _make_span(span_id=0x2222)
        bad_span.to_json.side_effect = Exception("serialization error")

        with patch("psycopg2.extras.execute_values") as mock_ev:
            self.exporter.export([good_span, bad_span])

        rows = mock_ev.call_args[0][2]
        self.assertEqual(len(rows), 1)

    @patch("psycopg2.connect")
    def test_reconnects_and_retries_on_operational_error(self, mock_connect):
        call_count = {"n": 0}

        def do_insert_side_effect(_rows):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise psycopg2.OperationalError("server closed connection")

        with patch.object(self.exporter, "_do_insert", side_effect=do_insert_side_effect):
            result = self.exporter.export([_make_span()])

        mock_connect.assert_called()
        self.assertEqual(call_count["n"], 2)
        self.assertEqual(result, SpanExportResult.SUCCESS)


class TestShutdown(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_POSTGRES_CONNECTION_URL"] = "postgresql://u:p@h/db"
        with patch("psycopg2.connect"):
            reload(pg_mod)
            self.exporter = pg_mod.PostgresSpanExporter()

    def tearDown(self):
        os.environ.pop("MONOCLE_POSTGRES_CONNECTION_URL", None)

    def test_shutdown_closes_connection(self):
        self.exporter.shutdown()
        self.exporter.connection.close.assert_called()


class TestEnsureTable(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_POSTGRES_CONNECTION_URL"] = "postgresql://u:p@h/db"

    def tearDown(self):
        os.environ.pop("MONOCLE_POSTGRES_CONNECTION_URL", None)

    def _make_mock_connection(self, cursor_execute_side_effect=None):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        if cursor_execute_side_effect:
            mock_cursor.execute.side_effect = cursor_execute_side_effect
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock_conn, mock_cursor

    @patch("psycopg2.connect")
    def test_create_table_called_on_init(self, mock_connect):
        mock_conn, mock_cursor = self._make_mock_connection()
        mock_connect.return_value = mock_conn

        reload(pg_mod)
        pg_mod.PostgresSpanExporter()

        mock_cursor.execute.assert_called_once()
        self.assertIn("CREATE TABLE IF NOT EXISTS traces",
                      mock_cursor.execute.call_args[0][0])
        mock_conn.commit.assert_called()

    @patch("psycopg2.connect")
    def test_permission_error_on_insufficient_privilege(self, mock_connect):
        mock_conn, _ = self._make_mock_connection(
            cursor_execute_side_effect=psycopg2.errors.InsufficientPrivilege("permission denied")  # pylint: disable=no-member
        )
        mock_connect.return_value = mock_conn

        reload(pg_mod)

        with self.assertRaises(PermissionError) as ctx:
            pg_mod.PostgresSpanExporter()
        self.assertIn("lacks CREATE TABLE permission", str(ctx.exception))
        mock_conn.rollback.assert_called_once()
