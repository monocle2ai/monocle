"""Unit tests for the trace-source abstraction and the Okahu test-result label.

Covers:
- ``get_trace_source`` name resolution.
- ``OkahuTraceSource.record_test_result`` request shape, PASS/FAIL mapping, and
  its best-effort gating / failure handling.
"""
import requests
from unittest.mock import MagicMock, patch

import pytest

from monocle_test_tools.trace_sources import (
    OkahuTraceSource,
    TraceSource,
    get_trace_source,
)


class _FakeOkahuSpanExporter:
    """Stand-in whose class name matches what the source detects."""


# Ensure the detected class name matches the real exporter's name.
_FakeOkahuSpanExporter.__name__ = "OkahuSpanExporter"


class _OtherExporter:
    pass


def _ok_response():
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.raise_for_status.return_value = None
    return response


def _record(source, monkeypatch, **overrides):
    """Call record_test_result with sensible defaults, returning (result, mock_post)."""
    monkeypatch.setenv("OKAHU_API_KEY", overrides.pop("api_key", "test-key"))
    kwargs = dict(
        fact_id="abc123",
        fact_name="traces",
        workflow_name="my_app",
        test_name="test_something",
        test_failed=False,
        exporters=[_FakeOkahuSpanExporter()],
    )
    kwargs.update(overrides)
    with patch("monocle_test_tools.trace_sources.okahu_trace_source.requests.post") as mock_post:
        mock_post.return_value = _ok_response()
        result = source.record_test_result(**kwargs)
    return result, mock_post


class TestGetTraceSource:
    def test_okahu_returns_okahu_source(self):
        assert isinstance(get_trace_source("okahu"), OkahuTraceSource)

    @pytest.mark.parametrize("name", ["file", "unknown", "", None])
    def test_other_returns_none(self, name):
        assert get_trace_source(name) is None


class TestBaseTraceSource:
    def test_default_record_is_noop(self):
        class Bare(TraceSource):
            name = "bare"

        assert Bare().record_test_result(
            fact_id="x", fact_name="traces", workflow_name="w",
            test_name="t", test_failed=False,
        ) is False


class TestOkahuRecordTestResult:
    def test_pass_posts_expected_request(self, monkeypatch):
        source = OkahuTraceSource()
        result, mock_post = _record(source, monkeypatch)

        assert result is True
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["url"].endswith("/v1/eval/label")
        assert kwargs["params"] == {
            "trace_id": "abc123",
            "fact_name": "traces",
            "workflow_name": "my_app",
        }
        assert kwargs["headers"] == {"x-api-key": "test-key"}
        assert kwargs["json"] == {
            "result": {
                "label": "test_something",
                "value": "PASS",
                "explanation": "PASS",
                "category": "test",
            }
        }

    def test_failed_maps_to_fail_value(self, monkeypatch):
        source = OkahuTraceSource()
        result, mock_post = _record(source, monkeypatch, test_failed=True)

        assert result is True
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["result"]["value"] == "FAIL"

    def test_custom_fact_passed_through(self, monkeypatch):
        source = OkahuTraceSource()
        _, mock_post = _record(source, monkeypatch, fact_name="agent_sessions", fact_id="sess-1")
        _, kwargs = mock_post.call_args
        assert kwargs["params"]["fact_name"] == "agent_sessions"
        assert kwargs["params"]["trace_id"] == "sess-1"

    def test_noop_without_okahu_exporter(self, monkeypatch):
        source = OkahuTraceSource()
        result, mock_post = _record(source, monkeypatch, exporters=[_OtherExporter()])
        assert result is False
        mock_post.assert_not_called()

    def test_noop_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OKAHU_API_KEY", raising=False)
        source = OkahuTraceSource()
        with patch("monocle_test_tools.trace_sources.okahu_trace_source.requests.post") as mock_post:
            result = source.record_test_result(
                fact_id="abc123", fact_name="traces", workflow_name="my_app",
                test_name="t", test_failed=False, exporters=[_FakeOkahuSpanExporter()],
            )
        assert result is False
        mock_post.assert_not_called()

    @pytest.mark.parametrize("missing", [{"fact_id": None}, {"workflow_name": None}])
    def test_noop_without_ids(self, monkeypatch, missing):
        source = OkahuTraceSource()
        result, mock_post = _record(source, monkeypatch, **missing)
        assert result is False
        mock_post.assert_not_called()

    def test_best_effort_on_network_error(self, monkeypatch):
        monkeypatch.setenv("OKAHU_API_KEY", "test-key")
        source = OkahuTraceSource()
        with patch("monocle_test_tools.trace_sources.okahu_trace_source.requests.post") as mock_post:
            mock_post.side_effect = requests.ConnectionError("boom")
            result = source.record_test_result(
                fact_id="abc123", fact_name="traces", workflow_name="my_app",
                test_name="t", test_failed=False, exporters=[_FakeOkahuSpanExporter()],
            )
        assert result is False

    def test_best_effort_on_non_2xx(self, monkeypatch):
        monkeypatch.setenv("OKAHU_API_KEY", "test-key")
        source = OkahuTraceSource()
        response = MagicMock(spec=requests.Response)
        response.raise_for_status.side_effect = requests.HTTPError("500")
        with patch("monocle_test_tools.trace_sources.okahu_trace_source.requests.post") as mock_post:
            mock_post.return_value = response
            result = source.record_test_result(
                fact_id="abc123", fact_name="traces", workflow_name="my_app",
                test_name="t", test_failed=False, exporters=[_FakeOkahuSpanExporter()],
            )
        assert result is False

    def test_endpoint_override(self, monkeypatch):
        monkeypatch.setenv("OKAHU_EVALUATION_ENDPOINT", "https://custom.example/api/")
        source = OkahuTraceSource()
        _, mock_post = _record(source, monkeypatch)
        _, kwargs = mock_post.call_args
        assert kwargs["url"] == "https://custom.example/api/v1/eval/label"
