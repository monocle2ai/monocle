"""Manual end-to-end test: make a real Azure OpenAI inference call, instrument
it with Monocle, export the spans to ClickHouse, and verify the workflow and
inference spans landed.

Run it manually (it hits Azure OpenAI and needs a live ClickHouse). Put the
settings in a ``.env`` at the repo root (or ``apptrace/.env``):

    MONOCLE_CLICKHOUSE_CONNECTION_URL=clickhouse://user:pass@host:8123/db
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
    AZURE_OPENAI_DEPLOYMENT_NAME=<your-deployment-name>   # or AZURE_OPENAI_API_DEPLOYMENT
    AZURE_OPENAI_API_VERSION=2024-10-21                   # optional, defaults below

Then:

    pytest apptrace/tests/integration/test_clickhouse_openai_manual.py -v -s

The test skips itself if the ClickHouse URL or the Azure settings are missing,
so it is safe in CI.
"""
import os
import logging

import pytest
from dotenv import load_dotenv

_HERE = os.path.dirname(__file__)
load_dotenv(os.path.join(_HERE, "..", "..", ".env"), override=True)         # apptrace/.env if present
load_dotenv(os.path.join(_HERE, "..", "..", "..", ".env"), override=True)   # repo-root .env wins

clickhouse_connect = pytest.importorskip("clickhouse_connect", reason="clickhouse-connect not installed")
openai = pytest.importorskip("openai", reason="openai not installed")

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.exporters.clickhouse.clickhouse_exporter import (
    ClickHouseSpanExporter,
    CLIENT_SETTINGS,
)

logger = logging.getLogger(__name__)

DEFAULT_API_VERSION = "2024-10-21"


def test_clickhouse_azure_openai_manual():
    conn_url = os.getenv("MONOCLE_CLICKHOUSE_CONNECTION_URL")
    if not conn_url:
        pytest.skip("MONOCLE_CLICKHOUSE_CONNECTION_URL not set")

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_API_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION
    if not (api_key and endpoint and deployment):
        pytest.skip("Azure OpenAI env vars not set (need API key, endpoint, deployment)")

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
        client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        response = client.chat.completions.create(
            model=deployment,  # Azure uses the deployment name
            messages=[{"role": "user", "content": "In one sentence, what is ClickHouse?"}],
        )
        logger.info("LLM answer: %s", response.choices[0].message.content)

        otel_trace.get_tracer_provider().force_flush()  # flush the batch processor to ClickHouse

        spans = memory_exporter.get_finished_spans()
        assert spans, "no spans were emitted"
        trace_id = f"0x{spans[0].context.trace_id:032x}"
        logger.info("trace_id: %s", trace_id)

        ch = clickhouse_connect.get_client(dsn=conn_url, settings=CLIENT_SETTINGS)
        try:
            span_types = [
                r[0] for r in ch.query(
                    "SELECT attributes.`span.type` FROM traces WHERE trace_id = {tid:String}",
                    parameters={"tid": trace_id},
                ).result_rows
            ]
        finally:
            ch.close()

        logger.info("span types in ClickHouse for %s: %s", trace_id, span_types)
        assert "workflow" in span_types, f"no workflow span in ClickHouse: {span_types}"
        assert any(t in ("inference", "inference.framework") for t in span_types), \
            f"no inference span in ClickHouse: {span_types}"
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
