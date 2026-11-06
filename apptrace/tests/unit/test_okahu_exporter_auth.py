"""A rejected Okahu API key logs one error, not one per span or exporter."""
import logging
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult

from monocle_apptrace.instrumentation.common.constants import MONOCLE_SDK_VERSION
from monocle_apptrace.exporters.okahu import okahu_exporter
from monocle_apptrace.exporters.okahu.okahu_exporter import OkahuSpanExporter


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(okahu_exporter, "get_monocle_env_value",
                        lambda key: "bad-key" if key == "OKAHU_API_KEY" else None)
    monkeypatch.setattr(OkahuSpanExporter, "_rejected_api_keys", {})


def _span():
    with TracerProvider().get_tracer("t").start_as_current_span("s", attributes={MONOCLE_SDK_VERSION: "1"}) as span:
        pass
    return span


def _session():
    session = MagicMock()
    session.headers = {}
    session.post.return_value = MagicMock(status_code=401, text="Authentication Failed")
    return session


def test_401_is_logged_once_per_endpoint(caplog):
    session = _session()
    exporters = [OkahuSpanExporter(session=session), OkahuSpanExporter(session=session),
                 OkahuSpanExporter(evaluate=True, session=session)]
    with caplog.at_level(logging.ERROR, logger=okahu_exporter.__name__):
        results = [exporter.export([_span()]) for exporter in exporters for _ in range(3)]
    assert results == [SpanExportResult.FAILURE] * 9
    assert all(exporter.last_status_code == 401 for exporter in exporters)
    # One request per endpoint: trace ingest and eval ingest.
    assert session.post.call_count == 2
    assert len(caplog.records) == 2


def test_401_is_retried_after_the_wait(monkeypatch):
    session = _session()
    exporter = OkahuSpanExporter(session=session)
    now = [1000.0]
    monkeypatch.setattr(okahu_exporter.time, "monotonic", lambda: now[0])
    exporter.export([_span()])
    exporter.export([_span()])
    assert session.post.call_count == 1
    now[0] += okahu_exporter.REJECTED_KEY_RETRY_SECONDS
    exporter.export([_span()])
    assert session.post.call_count == 2
