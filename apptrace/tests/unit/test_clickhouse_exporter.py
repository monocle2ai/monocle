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

# Column positions in the row built by _build_row (mirror of INSERT_COLUMNS).
C_NAME, C_START, C_END, C_STATUS_CODE, C_STATUS_MSG, \
    C_SPAN_ID, C_TRACE_ID, C_PARENT_ID, C_ATTRS, C_EVENTS, C_METADATA = range(11)


def _make_span(name="test-span", trace_id=0xAABBCCDD, span_id=0x1111,
               parent_span_id=None, status=None, attributes=None, events=None):
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
        "status": status if status is not None else {"status_code": "OK"},
        "attributes": attributes if attributes is not None else {"monocle_apptrace.version": "0.8.0"},
        "events": events if events is not None else [],
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
        mock_get_client.assert_called_once_with(
            dsn="clickhouse://user:pass@localhost:8123/db", settings=ch_mod.CLIENT_SETTINGS)
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
        self.assertEqual(row[C_SPAN_ID], "0xabcdef0123456789")

    def test_trace_id_has_0x_prefix(self):
        row = self.exporter._build_row(_make_span(trace_id=0xAABBCCDD11223344AABBCCDD11223344))
        self.assertTrue(row[C_TRACE_ID].startswith("0x"))

    def test_parent_id_is_none_for_root_span(self):
        row = self.exporter._build_row(_make_span())  # no parent
        self.assertIsNone(row[C_PARENT_ID])

    def test_parent_id_has_0x_prefix_for_child_span(self):
        row = self.exporter._build_row(_make_span(parent_span_id=0x1111222233334444))
        self.assertEqual(row[C_PARENT_ID], "0x1111222233334444")

    def test_timestamps_are_timezone_aware_datetimes(self):
        row = self.exporter._build_row(_make_span())
        self.assertIsInstance(row[C_START], datetime.datetime)
        self.assertIsInstance(row[C_END], datetime.datetime)
        self.assertIsNotNone(row[C_START].tzinfo)

    def test_status_split_into_code_and_message(self):
        row = self.exporter._build_row(
            _make_span(status={"status_code": "ERROR", "message": "boom"}))
        self.assertEqual(row[C_STATUS_CODE], "ERROR")
        self.assertEqual(row[C_STATUS_MSG], "boom")

    def test_status_message_none_when_absent(self):
        row = self.exporter._build_row(_make_span(status={"status_code": "OK"}))
        self.assertEqual(row[C_STATUS_CODE], "OK")
        self.assertIsNone(row[C_STATUS_MSG])

    def test_attributes_is_dict_for_json_column(self):
        row = self.exporter._build_row(
            _make_span(attributes={"span.type": "inference"}))
        self.assertEqual(row[C_ATTRS], {"span.type": "inference"})

    def test_events_is_list_for_array_json_column(self):
        events = [{"name": "metadata", "attributes": {"total_tokens": 42}}]
        row = self.exporter._build_row(_make_span(events=events))
        self.assertEqual(row[C_EVENTS], events)

    def test_metadata_is_empty_dict_reserved_column(self):
        # metadata column mirrors Postgres: present but reserved (empty) for now.
        row = self.exporter._build_row(_make_span())
        self.assertEqual(row[C_METADATA], {})

    def test_name_is_first_column(self):
        row = self.exporter._build_row(_make_span(name="my-span"))
        self.assertEqual(row[C_NAME], "my-span")


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
