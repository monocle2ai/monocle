"""End-to-end check that ClickHouseSpanExporter writes spans into the new
schema and that the data is queryable there:

- attributes  -> JSON        (sub-fields addressable, e.g. attributes.`span.type`)
- events      -> Array(JSON)  (searchable per element, e.g. by event name)
- status      -> status_code + status_message columns

The stored shape intentionally differs from the local ``.monocle`` file (the
JSON type re-nests dotted keys and status is split into columns), so this test
verifies the ClickHouse content directly rather than byte-for-byte parity.

Requires a running ClickHouse server reachable via
``MONOCLE_CLICKHOUSE_CONNECTION_URL`` (a single-binary ``clickhouse server`` is
enough); the test skips itself otherwise.
"""
import os
import logging

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect", reason="clickhouse-connect not installed")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from monocle_apptrace.exporters.clickhouse.clickhouse_exporter import (
    ClickHouseSpanExporter,
    CLIENT_SETTINGS,
)
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SDK_VERSION
from monocle_apptrace.instrumentation.common.utils import get_monocle_version

logger = logging.getLogger(__name__)


def test_clickhouse_schema_roundtrip():
    conn_url = os.getenv("MONOCLE_CLICKHOUSE_CONNECTION_URL")
    if not conn_url:
        pytest.skip("MONOCLE_CLICKHOUSE_CONNECTION_URL not set")

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ClickHouseSpanExporter()))
    tracer = provider.get_tracer(__name__)

    with tracer.start_as_current_span("clickhouse.workflow") as root:
        root.set_attribute(MONOCLE_SDK_VERSION, get_monocle_version())
        trace_hex = f"0x{root.get_span_context().trace_id:032x}"
        root_span_id = f"0x{root.get_span_context().span_id:016x}"
        with tracer.start_as_current_span("clickhouse.child") as child:
            child.set_attribute(MONOCLE_SDK_VERSION, get_monocle_version())
            child.set_attribute("span.type", "inference")
            child.add_event("metadata", {"total_tokens": 123})

    provider.shutdown()  # flushes the exporter

    client = clickhouse_connect.get_client(dsn=conn_url, settings=CLIENT_SETTINGS)
    try:
        rows = client.query(
            "SELECT name, span_id, parent_id, status_code "
            "FROM traces WHERE trace_id = {tid:String} ORDER BY name",
            parameters={"tid": trace_hex},
        ).result_rows
        by_name = {r[0]: r for r in rows}

        # both spans landed, with the parent/child link intact
        assert set(by_name) == {"clickhouse.workflow", "clickhouse.child"}, by_name
        assert by_name["clickhouse.workflow"][2] is None            # root has no parent
        assert by_name["clickhouse.child"][2] == root_span_id       # child -> root
        # status split into its own column (unset by default)
        assert by_name["clickhouse.workflow"][3] == "UNSET"

        # attributes JSON is queryable by sub-field
        span_type = client.query(
            "SELECT attributes.`span.type` FROM traces "
            "WHERE trace_id = {tid:String} AND name = 'clickhouse.child'",
            parameters={"tid": trace_hex},
        ).result_rows[0][0]
        assert span_type == "inference", span_type

        # events Array(JSON) is searchable per element
        has_metadata = client.query(
            "SELECT arrayExists(x -> x.name = 'metadata', events) FROM traces "
            "WHERE trace_id = {tid:String} AND name = 'clickhouse.child'",
            parameters={"tid": trace_hex},
        ).result_rows[0][0]
        assert has_metadata == 1
    finally:
        client.close()

    logger.info("Verified new-schema roundtrip in ClickHouse for trace %s", trace_hex)
