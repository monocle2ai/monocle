"""End-to-end check that ClickHouseSpanExporter persists the same span data
that FileSpanExporter writes to the local ``.monocle`` folder.

The same spans are exported through both exporters, then the rows read back
from ClickHouse are compared field-by-field against the JSON written to disk.

Requires a running ClickHouse server reachable via
``MONOCLE_CLICKHOUSE_CONNECTION_URL`` (a single-binary ``clickhouse server``
is enough); the test skips itself otherwise.
"""
import os
import glob
import json
import logging

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect", reason="clickhouse-connect not installed")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from monocle_apptrace.exporters.clickhouse.clickhouse_exporter import ClickHouseSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SDK_VERSION

logger = logging.getLogger(__name__)

# Fields both exporters persist and that must match between ClickHouse and .monocle.
JSON_FIELDS = ("status", "attributes", "events")


def _read_monocle_spans(out_dir: str, trace_hex: str) -> dict:
    """Return {span_id: serialized_span} for the given trace from the .monocle files."""
    spans = {}
    for file_path in glob.glob(os.path.join(out_dir, "*.json")):
        with open(file_path, encoding="UTF-8") as handle:
            for obj in json.load(handle):
                if obj["context"]["trace_id"] == trace_hex:
                    spans[obj["context"]["span_id"]] = obj
    return spans


def _read_clickhouse_spans(client, trace_hex: str) -> dict:
    """Return {span_id: row_dict} for the given trace from the traces table."""
    result = client.query(
        "SELECT name, span_id, parent_id, status, attributes, events "
        "FROM traces WHERE trace_id = {tid:String}",
        parameters={"tid": trace_hex},
    )
    spans = {}
    for name, span_id, parent_id, status, attributes, events in result.result_rows:
        spans[span_id] = {
            "name": name,
            "parent_id": parent_id,
            "status": json.loads(status),
            "attributes": json.loads(attributes),
            "events": json.loads(events),
        }
    return spans


def test_clickhouse_matches_monocle_file(tmp_path):
    conn_url = os.getenv("MONOCLE_CLICKHOUSE_CONNECTION_URL")
    if not conn_url:
        pytest.skip("MONOCLE_CLICKHOUSE_CONNECTION_URL not set")

    out_dir = str(tmp_path / ".monocle")
    file_exporter = FileSpanExporter(service_name="clickhouse_app", out_path=out_dir)
    ch_exporter = ClickHouseSpanExporter()

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(file_exporter))
    provider.add_span_processor(SimpleSpanProcessor(ch_exporter))
    tracer = provider.get_tracer(__name__)

    with tracer.start_as_current_span("clickhouse.workflow") as root:
        root.set_attribute(MONOCLE_SDK_VERSION, "0.8.0")
        root.set_attribute("entity.1.name", "clickhouse_app")
        trace_hex = f"0x{root.get_span_context().trace_id:032x}"
        with tracer.start_as_current_span("clickhouse.child") as child:
            child.set_attribute(MONOCLE_SDK_VERSION, "0.8.0")
            child.set_attribute("span.type", "inference")

    provider.shutdown()  # flushes both exporters and finalizes the .monocle file

    file_spans = _read_monocle_spans(out_dir, trace_hex)
    query_client = clickhouse_connect.get_client(dsn=conn_url)
    try:
        ch_spans = _read_clickhouse_spans(query_client, trace_hex)
    finally:
        query_client.close()

    assert file_spans, f"No spans found in .monocle for trace {trace_hex}"
    assert set(ch_spans) == set(file_spans), (
        f"span_id mismatch: clickhouse={sorted(ch_spans)} monocle={sorted(file_spans)}"
    )

    for span_id, file_obj in file_spans.items():
        ch_obj = ch_spans[span_id]
        assert ch_obj["name"] == file_obj["name"], f"name mismatch for {span_id}"
        assert ch_obj["parent_id"] == file_obj["parent_id"], f"parent_id mismatch for {span_id}"
        for field in JSON_FIELDS:
            assert ch_obj[field] == file_obj[field], f"{field} mismatch for {span_id}"

    logger.info(
        "Verified %s span(s) match between ClickHouse and .monocle for trace %s",
        len(ch_spans), trace_hex,
    )
