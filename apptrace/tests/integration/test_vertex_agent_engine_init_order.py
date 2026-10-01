"""An ADK agent under ``adk api_server --otel_to_cloud``, as ``adk deploy agent_engine`` runs it.

Checks that Monocle starts after Google's tracer provider, shares it, is flushed
by ``async_stream_query``, and labels the workflow span with the Agent Engine
host. Google's exporter is stubbed; the model call is real, so it needs
application-default credentials and GOOGLE_CLOUD_PROJECT.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once

try:
    from fastapi.testclient import TestClient
    from google.adk.cli.fast_api import get_fast_api_app
    from google.adk.telemetry.setup import OTelHooks
    from vertexai.agent_engines import AdkApp  # noqa: F401  (the server's query routes use it)

    AGENT_ENGINE_AVAILABLE = True
except ImportError:
    AGENT_ENGINE_AVAILABLE = False

AGENTS_DIR = Path(__file__).resolve().parent / "servers" / "vertex_agent_engine"
APP_NAME = "monocle_adk_agent"

# Hosting metadata only; the model call below uses the real GOOGLE_CLOUD_PROJECT.
ENGINE_ID = "1234567890123456789"
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = "us-central1"

# A GOOGLE_APPLICATION_CREDENTIALS pointing at a missing file hides working ADC.
_credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
BROKEN_CREDENTIALS_FILE = bool(_credentials_file) and not os.path.exists(_credentials_file)


def _adc_available() -> bool:
    import google.auth

    with patch.dict(os.environ):
        if BROKEN_CREDENTIALS_FILE:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        try:
            google.auth.default()
            return True
        except Exception:
            return False


pytestmark = pytest.mark.skipif(
    not (AGENT_ENGINE_AVAILABLE and PROJECT and _adc_available()),
    reason="google-adk with google-cloud-aiplatform[adk,agent_engines], GOOGLE_CLOUD_PROJECT "
           "or application-default credentials missing",
)


def _reset_otel():
    """Start from no global provider, as a fresh Agent Engine process does."""
    from monocle_apptrace.instrumentation.common.instrumentor import get_monocle_instrumentor

    instrumentor = get_monocle_instrumentor()
    if instrumentor is not None and instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()


def _unload_agent():
    for name in [m for m in sys.modules if m == APP_NAME or m.startswith(APP_NAME + ".")]:
        del sys.modules[name]


@pytest.fixture
def agent_engine_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", ENGINE_ID)
    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION", LOCATION)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", LOCATION)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", PROJECT)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    # Set by --otel_to_cloud; turns on the per-request flush.
    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", "true")
    # Keep the agent loader from loading the repo's .env and its hosting markers.
    monkeypatch.setenv("ADK_DISABLE_LOAD_DOTENV", "true")
    if BROKEN_CREDENTIALS_FILE:
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")
    saved_path = list(sys.path)
    _unload_agent()
    _reset_otel()
    yield
    _reset_otel()
    _unload_agent()
    sys.path[:] = saved_path


def _processor_exporters(provider):
    """Exporter of each batch processor on the provider's active processor."""
    exporters = []
    for processor in provider._active_span_processor._span_processors:
        exporter = getattr(processor, "span_exporter", None) or getattr(
            getattr(processor, "_batch_processor", None), "_exporter", None)
        if exporter is not None:
            exporters.append(exporter)
    return exporters


def test_monocle_starts_after_google_and_shares_its_provider(agent_engine_env):
    import monocle_apptrace

    # Both processors hold spans for 10 minutes unless flushed.
    google_exporter = InMemorySpanExporter()
    google_processor = BatchSpanProcessor(google_exporter, schedule_delay_millis=600_000)
    monocle_exporter = InMemorySpanExporter()
    monocle_processor = BatchSpanProcessor(monocle_exporter, schedule_delay_millis=600_000)

    real_setup = monocle_apptrace.setup_monocle_telemetry
    provider_at_monocle_setup = []

    def setup_with_test_processor(workflow_name=None, **_):
        provider_at_monocle_setup.append(trace.get_tracer_provider())
        return real_setup(workflow_name=workflow_name, span_processors=[monocle_processor])

    with patch("google.adk.telemetry.google_cloud.get_gcp_exporters",
               lambda **_: OTelHooks(span_processors=[google_processor])), \
            patch.object(monocle_apptrace, "setup_monocle_telemetry", setup_with_test_processor):
        # The server adk deploy's Dockerfile starts.
        app = get_fast_api_app(
            agents_dir=str(AGENTS_DIR),
            web=False,
            otel_to_cloud=True,
            session_service_uri="memory://",
            artifact_service_uri="memory://",
            memory_service_uri="memory://",
            use_local_storage=False,
            gemini_enterprise_app_name=APP_NAME,
        )

        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider), "the server should have installed an SDK provider"
        assert google_processor in provider._active_span_processor._span_processors
        assert f"{APP_NAME}.agent" not in sys.modules, "the server imported the agent at startup"
        assert not provider_at_monocle_setup, "Monocle was set up before the first request"

        with TestClient(app) as client:
            response = client.post("/api/stream_reasoning_engine", json={
                "class_method": "async_stream_query",
                "input": {"message": "Book a flight from BOM to JFK.", "user_id": "test_user"},
            })
            assert response.status_code == 200, response.text
            events = [json.loads(line) for line in response.text.splitlines() if line.strip()]

            assert events, "the agent returned no events"
            assert provider_at_monocle_setup == [provider], (
                "Monocle should be set up once, on the first request, with Google's provider in place")
            assert monocle_processor in provider._active_span_processor._span_processors, (
                "Monocle's processor must be attached to Google's provider, not a provider of its own")
            assert google_exporter in _processor_exporters(provider), "Google's processor must still be installed"

            # Read before shutdown, so only the per-request flush can have exported these.
            spans = monocle_exporter.get_finished_spans()
            assert spans, "async_stream_query's flush did not reach Monocle's processor"
            google_spans = google_exporter.get_finished_spans()
            assert google_spans, "Google's processor received no spans"

            span_types = [str(s.attributes.get("span.type", "")) for s in spans]
            workflow = [s for s in spans if s.attributes.get("span.type") == "workflow"]
            assert len(workflow) == 1, f"expected one workflow span, got {span_types}"
            assert workflow[0].attributes.get("entity.2.type") == "app_hosting.gcp_agent_engine"
            assert workflow[0].attributes.get("entity.2.name") == (
                f"projects/{os.environ['GOOGLE_CLOUD_PROJECT']}/locations/{LOCATION}/reasoningEngines/{ENGINE_ID}")
            assert any(t.startswith("inference") for t in span_types), f"no inference span in {span_types}"
            assert "agentic.tool.invocation" in span_types, f"no tool span in {span_types}"
            # Both processors sit on one provider, so they see the same trace.
            assert {s.context.trace_id for s in spans} & {s.context.trace_id for s in google_spans}
