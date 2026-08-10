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

# clickhouse-connect is an optional dependency (installed via the `clickhouse` extra).
clickhouse_connect = pytest.importorskip("clickhouse_connect")
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
import monocle_apptrace.exporters.clickhouse.clickhouse_exporter as ch_mod
from monocle_apptrace.exporters.clickhouse.clickhouse_exporter import ClickHouseSpanExporter
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


class TestClickHouseRegistry(unittest.TestCase):
    def test_clickhouse_in_registry(self):
        self.assertIn("clickhouse", monocle_exporters)
        entry = monocle_exporters["clickhouse"]
        self.assertEqual(entry["module"], "monocle_apptrace.exporters.clickhouse.clickhouse_exporter")
        self.assertEqual(entry["class"], "ClickHouseSpanExporter")


class TestClickHouseInit(unittest.TestCase):
    @patch("clickhouse_connect.get_client")
    def test_reads_connection_url_from_env(self, mock_get_client):
        os.environ["MONOCLE_CLICKHOUSE_CONNECTION_URL"] = "clickhouse://user:pass@localhost:8123/db"
        exporter = ClickHouseSpanExporter()
        mock_get_client.assert_called_once_with(dsn="clickhouse://user:pass@localhost:8123/db")
        self.assertIsNotNone(exporter.client)
        del os.environ["MONOCLE_CLICKHOUSE_CONNECTION_URL"]

    @patch("clickhouse_connect.get_client")
    def test_raises_when_url_missing(self, mock_get_client):
        os.environ.pop("MONOCLE_CLICKHOUSE_CONNECTION_URL", None)
        with self.assertRaises(ValueError):
            ClickHouseSpanExporter()
        mock_get_client.assert_not_called()


class TestBuildRow(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_CLICKHOUSE_CONNECTION_URL"] = "clickhouse://u:p@h:8123/db"
        with patch("clickhouse_connect.get_client"):
            reload(ch_mod)
            self.exporter = ch_mod.ClickHouseSpanExporter()

    def tearDown(self):
        os.environ.pop("MONOCLE_CLICKHOUSE_CONNECTION_URL", None)

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

    def test_json_columns_are_serialized_strings(self):
        row = self.exporter._build_row(_make_span())
        self.assertEqual(json.loads(row[3]), {"status_code": "OK"})
        self.assertEqual(json.loads(row[7]), {"monocle_apptrace.version": "0.8.0"})
        self.assertEqual(json.loads(row[8]), [])

    def test_metadata_is_none(self):
        row = self.exporter._build_row(_make_span())
        self.assertIsNone(row[9])


class TestExport(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_CLICKHOUSE_CONNECTION_URL"] = "clickhouse://u:p@h:8123/db"
        with patch("clickhouse_connect.get_client"):
            reload(ch_mod)
            self.exporter = ch_mod.ClickHouseSpanExporter()

    def tearDown(self):
        os.environ.pop("MONOCLE_CLICKHOUSE_CONNECTION_URL", None)

    def test_export_calls_insert_once_per_batch(self):
        child = _make_span(span_id=0x2222, parent_span_id=0x1111)
        root = _make_span(span_id=0x1111)

        result = self.exporter.export([child, root])

        self.assertEqual(result, SpanExportResult.SUCCESS)
        self.exporter.client.insert.assert_called_once()
        args, kwargs = self.exporter.client.insert.call_args
        self.assertEqual(args[0], "traces")
        self.assertEqual(len(args[1]), 2)
        self.assertEqual(kwargs["column_names"], ch_mod.INSERT_COLUMNS)

    def test_export_does_not_insert_when_no_rows(self):
        span = _make_span()
        span.attributes.get.return_value = None  # triggers skip_export

        result = self.exporter.export([span])

        self.assertEqual(result, SpanExportResult.SUCCESS)
        self.exporter.client.insert.assert_not_called()

    def test_export_returns_failure_on_exception(self):
        with patch.object(self.exporter, "skip_export", side_effect=RuntimeError("boom")):
            result = self.exporter.export([_make_span()])
        self.assertEqual(result, SpanExportResult.FAILURE)

    def test_bad_span_skipped_good_span_inserted(self):
        good_span = _make_span(span_id=0x1111)
        bad_span = _make_span(span_id=0x2222)
        bad_span.to_json.side_effect = Exception("serialization error")

        self.exporter.export([good_span, bad_span])

        rows = self.exporter.client.insert.call_args[0][1]
        self.assertEqual(len(rows), 1)

    def test_reconnects_and_retries_on_operational_error(self):
        call_count = {"n": 0}

        def do_insert_side_effect(_rows):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OperationalError("server closed connection")

        with patch("clickhouse_connect.get_client") as mock_get_client, \
                patch.object(self.exporter, "_do_insert", side_effect=do_insert_side_effect):
            result = self.exporter.export([_make_span()])

        mock_get_client.assert_called()
        self.assertEqual(call_count["n"], 2)
        self.assertEqual(result, SpanExportResult.SUCCESS)


class TestShutdown(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_CLICKHOUSE_CONNECTION_URL"] = "clickhouse://u:p@h:8123/db"
        with patch("clickhouse_connect.get_client"):
            reload(ch_mod)
            self.exporter = ch_mod.ClickHouseSpanExporter()

    def tearDown(self):
        os.environ.pop("MONOCLE_CLICKHOUSE_CONNECTION_URL", None)

    def test_shutdown_closes_client(self):
        self.exporter.shutdown()
        self.exporter.client.close.assert_called()


class TestEnsureTable(unittest.TestCase):
    def setUp(self):
        os.environ["MONOCLE_CLICKHOUSE_CONNECTION_URL"] = "clickhouse://u:p@h:8123/db"

    def tearDown(self):
        os.environ.pop("MONOCLE_CLICKHOUSE_CONNECTION_URL", None)

    @patch("clickhouse_connect.get_client")
    def test_create_table_called_on_init(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        reload(ch_mod)
        ch_mod.ClickHouseSpanExporter()

        mock_client.command.assert_called_once()
        self.assertIn("CREATE TABLE IF NOT EXISTS traces",
                      mock_client.command.call_args[0][0])

    @patch("clickhouse_connect.get_client")
    def test_permission_error_on_access_denied(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.command.side_effect = DatabaseError("Code: 497. DB::Exception: ACCESS_DENIED")
        mock_get_client.return_value = mock_client

        reload(ch_mod)

        with self.assertRaises(PermissionError) as ctx:
            ch_mod.ClickHouseSpanExporter()
        self.assertIn("lacks CREATE TABLE permission", str(ctx.exception))
