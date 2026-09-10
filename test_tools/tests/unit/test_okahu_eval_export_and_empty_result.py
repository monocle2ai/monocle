"""Backend failures must be reported as what they are (issue #816).

A failed trace export used to be swallowed and the trace marked as exported,
and an empty ``result`` list was reported as a malformed response.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from monocle_apptrace.exporters.okahu.okahu_eval_result_exporter import OkahuEvalResultExporter
from monocle_test_tools.evals import okahu_eval as okahu_eval_module
from monocle_test_tools.evals.okahu_eval import OkahuEval


def _eval() -> OkahuEval:
    return OkahuEval(eval_options={"trace_source": "file"})


def _span():
    span = MagicMock()
    span.get_span_context.return_value.trace_id = 0xABC
    span.attributes = {"workflow.name": "wf"}
    span.start_time = 1_000_000_000
    span.end_time = 2_000_000_000
    return span


def _exporter(result, status_code=None):
    exporter = MagicMock()
    exporter.export.return_value = result
    exporter.last_status_code = status_code
    return exporter


def _resp(body: dict):
    m = MagicMock()
    m.headers = {"Content-Type": "application/json"}
    m.raise_for_status.return_value = None
    m.json.return_value = body
    return m


class TestExportTrace:
    def test_failed_export_raises_with_status_and_stays_unexported(self, monkeypatch):
        monkeypatch.setenv("OKAHU_API_KEY", "k")
        ev = _eval()
        exporter = _exporter(SpanExportResult.FAILURE, status_code=503)
        with patch.object(okahu_eval_module.okahu_exporter, "OkahuSpanExporter", return_value=exporter):
            with pytest.raises(AssertionError) as info:
                ev.export_trace([_span()])
        assert "HTTP 503" in str(info.value)
        assert ev._trace_exported is False
        exporter.shutdown.assert_called_once()

    def test_failed_export_without_response_names_the_timeout(self, monkeypatch):
        monkeypatch.setenv("OKAHU_API_KEY", "k")
        ev = _eval()
        exporter = _exporter(SpanExportResult.FAILURE, status_code=None)
        with patch.object(okahu_eval_module.okahu_exporter, "OkahuSpanExporter", return_value=exporter):
            with pytest.raises(AssertionError, match="no HTTP response"):
                ev.export_trace([_span()])
        assert ev._trace_exported is False

    def test_nothing_uploaded_raises(self, monkeypatch):
        monkeypatch.setenv("OKAHU_API_KEY", "k")
        ev = _eval()
        exporter = _exporter(None)
        with patch.object(okahu_eval_module.okahu_exporter, "OkahuSpanExporter", return_value=exporter):
            with pytest.raises(AssertionError, match="filtered out"):
                ev.export_trace([_span()])
        assert ev._trace_exported is False

    def test_successful_export_marks_trace_exported(self, monkeypatch):
        monkeypatch.setenv("OKAHU_API_KEY", "k")
        ev = _eval()
        exporter = _exporter(SpanExportResult.SUCCESS, status_code=202)
        with patch.object(okahu_eval_module.okahu_exporter, "OkahuSpanExporter", return_value=exporter):
            trace_id = ev.export_trace([_span()])
        assert trace_id == format(0xABC, "032x")
        assert ev._trace_exported is True


def _run_evaluate(ev, body):
    with patch.object(OkahuEval, "export_trace", return_value="traceid"), \
         patch.object(OkahuEval, "enumerate_fact_ids", return_value=["traceid"]), \
         patch("monocle_test_tools.evals.okahu_eval.requests.post", return_value=_resp(body)), \
         patch.object(OkahuEvalResultExporter, "export_results", return_value=None):
        return ev.evaluate(filtered_spans=[_span()], template={"name": "t"}, fact_name="traces")


def test_empty_result_is_reported_as_no_results(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "k")
    with pytest.raises(AssertionError) as info:
        _run_evaluate(_eval(), {"job_id": "job-1", "result": []})
    message = str(info.value)
    assert "no results" in message
    assert "job-1" in message
    assert "Expected 'result' key" not in message


def test_malformed_result_entry_is_still_a_format_error(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "k")
    with pytest.raises(AssertionError, match="Unexpected response format"):
        _run_evaluate(_eval(), {"job_id": "job-1", "result": [{"result": "not json"}]})


def test_empty_time_pad_env_uses_the_default(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "k")
    monkeypatch.setenv("OKAHU_EVAL_TIME_PAD_SECONDS", "")
    judge = {"label": "ok", "explanation": "fine"}
    label, explanation = _run_evaluate(_eval(), {"job_id": "job-1", "result": [{"result": json.dumps(judge)}]})
    assert (label, explanation) == ("ok", "fine")
