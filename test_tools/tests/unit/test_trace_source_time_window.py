"""Unit tests for the start_time/end_time time-window support on the trace-source APIs.

Covers:
- ``JSONSpanLoader.from_json`` rejecting start_time/end_time (unsupported for files).
- ``OkahuSpanLoader`` passing start_time/end_time through to the Okahu HTTP APIs.
- ``import_traces`` threading the window to the loaders (okahu id-bound lookup, file reject).
- ``with_trace_source("okahu", id=..., start_time=..., end_time=...)`` importing (not filter mode).

No live HTTP is performed; the low-level GET helper is mocked to capture params.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from monocle_test_tools import MonocleValidator
from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.file_span_loader import JSONSpanLoader
from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

START = "2026-07-23T19:29:22.846181Z"
END = "2026-07-23T20:29:22.846181Z"


@pytest.fixture(autouse=True)
def _reset_trace_assertion_class_state():
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None


# --------------------------------------------------------------------------- #
# JSONSpanLoader rejects a time window
# --------------------------------------------------------------------------- #

def test_json_span_loader_rejects_start_time(tmp_path):
    f = tmp_path / "trace.json"
    f.write_text("[]")
    with pytest.raises(ValueError, match="start_time"):
        JSONSpanLoader.from_json(str(f), start_time=START)


def test_json_span_loader_rejects_end_time(tmp_path):
    f = tmp_path / "trace.json"
    f.write_text("[]")
    with pytest.raises(ValueError, match="end_time"):
        JSONSpanLoader.from_json(str(f), end_time=END)


def test_json_span_loader_no_window_still_loads(tmp_path):
    f = tmp_path / "trace.json"
    f.write_text("[]")
    assert JSONSpanLoader.from_json(str(f)) == []


# --------------------------------------------------------------------------- #
# OkahuSpanLoader passes the window to the HTTP layer
# --------------------------------------------------------------------------- #

def _mock_env(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "test-key")
    monkeypatch.delenv("OKAHU_API_ENDPOINT", raising=False)


def test_get_trace_ids_passes_window(monkeypatch):
    _mock_env(monkeypatch)
    with patch.object(OkahuSpanLoader, "_do_get", return_value=[]) as mock_get:
        OkahuSpanLoader.get_trace_ids("wf", "agent_sessions", "sess_1",
                                      start_time=START, end_time=END)
    params = mock_get.call_args.kwargs["params"]
    assert params["start_time"] == START
    assert params["end_time"] == END


def test_get_spans_passes_window(monkeypatch):
    _mock_env(monkeypatch)
    workflow_span = MagicMock()
    workflow_span.attributes = {"span.type": "workflow"}
    with patch.object(JSONSpanLoader, "_from_dict", return_value=workflow_span), \
         patch.object(OkahuSpanLoader, "_do_get", return_value=[{}]) as mock_get:
        OkahuSpanLoader.get_spans("wf", "abc123", start_time=START, end_time=END)
    params = mock_get.call_args.kwargs["params"]
    assert params["start_time"] == START
    assert params["end_time"] == END


def test_get_spans_no_window_omits_params(monkeypatch):
    _mock_env(monkeypatch)
    workflow_span = MagicMock()
    workflow_span.attributes = {"span.type": "workflow"}
    with patch.object(JSONSpanLoader, "_from_dict", return_value=workflow_span), \
         patch.object(OkahuSpanLoader, "_do_get", return_value=[{}]) as mock_get:
        OkahuSpanLoader.get_spans("wf", "abc123")
    # No filter/window -> params is None (unchanged behavior).
    assert mock_get.call_args.kwargs["params"] is None


def test_load_by_scope_threads_window(monkeypatch):
    _mock_env(monkeypatch)
    with patch.object(OkahuSpanLoader, "get_trace_ids", return_value=["t1"]) as mock_ids, \
         patch.object(OkahuSpanLoader, "get_spans", return_value=[]) as mock_spans:
        OkahuSpanLoader.load_by_scope("wf", "agent_sessions", "sess_1",
                                      start_time=START, end_time=END)
    assert mock_ids.call_args.kwargs["start_time"] == START
    assert mock_ids.call_args.kwargs["end_time"] == END
    assert mock_spans.call_args.kwargs["start_time"] == START
    assert mock_spans.call_args.kwargs["end_time"] == END


# --------------------------------------------------------------------------- #
# import_traces threads the window to the loaders
# --------------------------------------------------------------------------- #

def test_import_traces_okahu_trace_passes_window():
    validator = MonocleValidator()
    with patch.object(OkahuSpanLoader, "get_spans", return_value=[]) as mock_spans:
        validator.import_traces(trace_source="okahu", id="abc123",
                                workflow_name="wf", fact_name="trace",
                                start_time=START, end_time=END)
    assert mock_spans.call_args.kwargs["start_time"] == START
    assert mock_spans.call_args.kwargs["end_time"] == END


def test_import_traces_okahu_scope_passes_window():
    validator = MonocleValidator()
    with patch.object(OkahuSpanLoader, "load_by_scope", return_value=[]) as mock_scope:
        validator.import_traces(trace_source="okahu", id="sess_1",
                                workflow_name="wf", fact_name="session",
                                start_time=START, end_time=END)
    assert mock_scope.call_args.kwargs["start_time"] == START
    assert mock_scope.call_args.kwargs["end_time"] == END


def test_import_traces_file_with_window_raises(tmp_path):
    trace_file = tmp_path / "monocle_trace_svc_deadbeef_20250101.json"
    trace_file.write_text("[]")
    validator = MonocleValidator()
    with pytest.raises(ValueError, match="start_time"):
        validator.import_traces(trace_source="file", trace_path=str(trace_file),
                                start_time=START)


# --------------------------------------------------------------------------- #
# with_trace_source: okahu id + window imports (bounded lookup, not filter mode)
# --------------------------------------------------------------------------- #

def test_with_trace_source_okahu_id_and_window_imports():
    a = TraceAssertion()
    with patch.object(a.validator, "import_traces") as imp:
        a.with_trace_source("okahu", id="t1", workflow_name="wf",
                            start_time=START, end_time=END)
    imp.assert_called_once()
    assert imp.call_args.kwargs["start_time"] == START
    assert imp.call_args.kwargs["end_time"] == END
    # id + window is a bounded lookup, NOT the eval-only filter mode.
    assert a._okahu_filter is None


def test_with_trace_source_okahu_window_only_still_filter_mode():
    a = TraceAssertion()
    with patch.object(a.validator, "import_traces") as imp:
        a.with_trace_source("okahu", workflow_name="wf",
                            start_time=START, end_time=END)
    imp.assert_not_called()
    assert a._okahu_filter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
