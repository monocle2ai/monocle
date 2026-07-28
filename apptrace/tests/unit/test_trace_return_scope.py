from monocle_apptrace.instrumentation.common.utils import extract_http_headers, get_scopes, clear_http_scopes


def test_scope_set_when_enabled_and_authorized(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    monkeypatch.delenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", raising=False)
    token = extract_http_headers({"x-monocle-retrieve-traces": "s3cret"})
    try:
        assert "monocle_trace_return" in get_scopes()
    finally:
        clear_http_scopes(token)


def test_scope_absent_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    token = extract_http_headers({"x-monocle-retrieve-traces": "s3cret"})
    try:
        assert "monocle_trace_return" not in get_scopes()
    finally:
        clear_http_scopes(token)


def test_scope_absent_when_not_authorized(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    monkeypatch.delenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", raising=False)
    token = extract_http_headers({"x-monocle-retrieve-traces": "wrong-key"})
    try:
        assert "monocle_trace_return" not in get_scopes()
    finally:
        clear_http_scopes(token)
