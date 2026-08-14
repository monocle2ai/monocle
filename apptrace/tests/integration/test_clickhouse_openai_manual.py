"""Manual end-to-end test: make a real OpenAI inference call, instrument it
with Monocle, export the spans to ClickHouse, and verify the workflow and
inference spans landed.

Run it manually (it hits the OpenAI API and needs a live ClickHouse). Put the
settings in ``apptrace/.env``:

    MONOCLE_CLICKHOUSE_CONNECTION_URL=clickhouse://user:pass@host:8123/db
    OPENAI_API_KEY=sk-...

Then:

    pytest apptrace/tests/integration/test_clickhouse_openai_manual.py -v -s

The test skips itself if either variable is missing, so it is safe in CI.
"""
import os
import logging

import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

clickhouse_connect = pytest.importorskip("clickhouse_connect", reason="clickhouse-connect not installed")
pytest.importorskip("openai", reason="openai not installed")

from openai import OpenAI
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.exporters.clickhouse.clickhouse_exporter import (
    ClickHouseSpanExporter,
    CLIENT_SETTINGS,
)

logger = logging.getLogger(__name__)


def test_clickhouse_openai_manual():
    conn_url = os.getenv("MONOCLE_CLICKHOUSE_CONNECTION_URL")
    if not conn_url:
        pytest.skip("MONOCLE_CLICKHOUSE_CONNECTION_URL not set")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Capture the emitted spans locally so we know the trace_id to query for.
    memory_exporter = InMemorySpanExporter()
    instrumentor = setup_monocle_telemetry(
        workflow_name="clickhouse_manual_test",
        span_processors=[
            BatchSpanProcessor(ClickHouseSpanExporter()),
            SimpleSpanProcessor(memory_exporter),
        ],
        wrapper_methods=[],
    )
    try:
        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "In one sentence, what is ClickHouse?"}],
        )
        logger.info("LLM answer: %s", response.choices[0].message.content)

        otel_trace.get_tracer_provider().force_flush()  # flush the batch processor to ClickHouse

        spans = memory_exporter.get_finished_spans()
        assert spans, "no spans were emitted"
        trace_id = f"0x{spans[0].context.trace_id:032x}"
        logger.info("trace_id: %s", trace_id)

        client = clickhouse_connect.get_client(dsn=conn_url, settings=CLIENT_SETTINGS)
        try:
            span_types = [
                r[0] for r in client.query(
                    "SELECT attributes.`span.type` FROM traces WHERE trace_id = {tid:String}",
                    parameters={"tid": trace_id},
                ).result_rows
            ]
        finally:
            client.close()

        logger.info("span types in ClickHouse for %s: %s", trace_id, span_types)
        assert "workflow" in span_types, f"no workflow span in ClickHouse: {span_types}"
        assert any(t in ("inference", "inference.framework") for t in span_types), \
            f"no inference span in ClickHouse: {span_types}"
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
