from opentelemetry.context import attach, detach
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, get_scopes, clear_http_scopes


def test_scope_set_when_enabled_and_requested(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    token = extract_http_headers({"x-monocle-retrieve-traces": "true"})
    try:
        assert "monocle_trace_return" in get_scopes()
    finally:
        clear_http_scopes(token)


def test_scope_absent_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    token = extract_http_headers({"x-monocle-retrieve-traces": "true"})
    try:
        assert "monocle_trace_return" not in get_scopes()
    finally:
        clear_http_scopes(token)


def test_scope_absent_when_not_requested(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    token = extract_http_headers({"other": "true"})
    try:
        assert "monocle_trace_return" not in get_scopes()
    finally:
        clear_http_scopes(token)
