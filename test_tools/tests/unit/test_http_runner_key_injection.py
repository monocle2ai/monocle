from monocle_test_tools.runner.http_runner import HttpRunner


def test_injects_key_when_env_set(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_KEY", "s3cret")
    runner = HttpRunner()
    kwargs = {"method": "POST"}
    runner._maybe_inject_retrieval_key(kwargs)
    assert kwargs["headers"]["x-monocle-retrieve-traces"] == "s3cret"


def test_caller_header_wins(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_KEY", "s3cret")
    runner = HttpRunner()
    kwargs = {"headers": {"X-Monocle-Retrieve-Traces": "caller-value"}}
    runner._maybe_inject_retrieval_key(kwargs)
    # caller-supplied value preserved (case-insensitive detection), key not overwritten
    values = {k.lower(): v for k, v in kwargs["headers"].items()}
    assert values["x-monocle-retrieve-traces"] == "caller-value"


def test_no_injection_when_env_unset(monkeypatch):
    monkeypatch.delenv("MONOCLE_TRACE_RETRIEVAL_KEY", raising=False)
    runner = HttpRunner()
    kwargs = {"method": "GET"}
    runner._maybe_inject_retrieval_key(kwargs)
    assert "headers" not in kwargs or all(
        k.lower() != "x-monocle-retrieve-traces" for k in kwargs.get("headers", {})
    )
