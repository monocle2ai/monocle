"""Unit tests for the eval-discovery unit."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from monocle_test_tools.evals import eval_discovery
from monocle_test_tools.evals.eval_discovery import _DiscoverySkipped, _derive_query_inputs


def _span(attributes, trace_id=0xABC123, start=None, end=None):
    return SimpleNamespace(
        attributes=attributes,
        events=[],
        start_time=start,
        end_time=end,
        get_span_context=lambda: SimpleNamespace(trace_id=trace_id),
    )


def test_derive_query_inputs_for_traces():
    spans = [
        _span({"span.type": "workflow", "workflow.name": "wf"},
              trace_id=0xABC123, start=1_000_000_000_000_000_000, end=1_000_000_002_000_000_000),
    ]
    # fact_map: mapped okahu fact "traces" -> id source "trace_id"
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.get_fact_map",
               return_value={"traces": "trace_id"}):
        wf, fact_ids, start, end = _derive_query_inputs(spans, "traces")
    assert wf == "wf"
    assert fact_ids == ["00000000000000000000000000abc123"]
    assert start < end  # padded window, ISO strings


def test_derive_query_inputs_missing_workflow_skips():
    spans = [_span({"span.type": "workflow"}, start=1, end=2)]
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.get_fact_map",
               return_value={"traces": "trace_id"}):
        with pytest.raises(_DiscoverySkipped):
            _derive_query_inputs(spans, "traces")


# --- discover_fact_evals orchestration -------------------------------------------

from monocle_test_tools.evals.eval_discovery import discover_fact_evals


def _labeled_row(fact_id, label, eval_name=None):
    return {"fact_id": fact_id, "eval_name": eval_name, "eval_found": True,
            "eval_result": {"label": label, "explanation": "x"}}


def _traces_spans():
    return [_span({"span.type": "workflow", "workflow.name": "wf"},
                  trace_id=0xABC123, start=1_000_000_000_000_000_000, end=1_000_000_002_000_000_000)]


def _set_okahu_env(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "k")
    monkeypatch.setenv("OKAHU_EVALUATION_ENDPOINT", "https://eval.example/api")
    monkeypatch.setenv("OKAHU_API_ENDPOINT", "https://api.example")


def test_discover_returns_specs_for_labeled_evals(monkeypatch):
    _set_okahu_env(monkeypatch)
    with patch("monocle_test_tools.evals.eval_discovery._derive_query_inputs",
               return_value=("wf", ["abc123"], "s", "e")), \
         patch("monocle_test_tools.evals.eval_discovery._list_candidate_evals",
               return_value=["correctness", "bias"]), \
         patch("monocle_test_tools.evals.eval_discovery._query_fact_evals",
               return_value=[
                   _labeled_row("abc123", "correct", eval_name="correctness"),
                   _labeled_row("abc123", "unbiased", eval_name="bias"),
               ]):
        specs, note = discover_fact_evals(_traces_spans(), fact_name="traces")

    assert note is None
    by_name = {s["criteria"]: s for s in specs}
    assert by_name["correctness"]["expected"] == "correct"
    assert by_name["correctness"]["fact_name"] == "traces"
    assert by_name["correctness"]["_discovered"] is True
    assert by_name["correctness"]["_discovered_fact_id"] == "abc123"
    assert by_name["bias"]["expected"] == "unbiased"


def test_discover_empty_when_no_labels(monkeypatch):
    _set_okahu_env(monkeypatch)
    with patch("monocle_test_tools.evals.eval_discovery._derive_query_inputs",
               return_value=("wf", ["abc123"], "s", "e")), \
         patch("monocle_test_tools.evals.eval_discovery._list_candidate_evals",
               return_value=["correctness"]), \
         patch("monocle_test_tools.evals.eval_discovery._query_fact_evals",
               return_value=[]):
        specs, note = discover_fact_evals(_traces_spans(), fact_name="traces")
    assert specs == []
    assert note == "No existing evals found on this fact"


def test_discover_unsupported_source_skips():
    specs, note = discover_fact_evals(_traces_spans(), fact_name="traces", eval_source="nope")
    assert specs == []
    assert note.startswith("eval discovery skipped:")


def test_discover_missing_creds_skips(monkeypatch):
    monkeypatch.delenv("OKAHU_API_KEY", raising=False)
    specs, note = discover_fact_evals(_traces_spans(), fact_name="traces")
    assert specs == []
    assert note.startswith("eval discovery skipped:")


def test_discover_http_error_is_non_fatal(monkeypatch):
    _set_okahu_env(monkeypatch)
    with patch("monocle_test_tools.evals.eval_discovery._derive_query_inputs",
               side_effect=RuntimeError("boom")):
        specs, note = discover_fact_evals(_traces_spans(), fact_name="traces")
    assert specs == []
    assert note.startswith("eval discovery skipped:")
