"""Unit tests for the eval-discovery unit."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from monocle_test_tools.evals import okahu_eval_discovery
from monocle_test_tools.evals.okahu_eval_discovery import _DiscoverySkipped, _derive_query_inputs


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
        wf, fact_ids, start_time, end_time = _derive_query_inputs(spans, "traces")
    assert wf == "wf"
    assert fact_ids == ["00000000000000000000000000abc123"]
    # Window is span-derived and padded outward: start < earliest, end > latest.
    assert start_time is not None and end_time is not None
    assert start_time.endswith("Z") and end_time.endswith("Z")
    assert start_time < "2001-09-09" and end_time > "2001-09-09"


def test_derive_query_inputs_no_span_times_omits_window():
    spans = [_span({"span.type": "workflow", "workflow.name": "wf"},
                   trace_id=0xABC123, start=None, end=None)]
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.get_fact_map",
               return_value={"traces": "trace_id"}):
        wf, fact_ids, start_time, end_time = _derive_query_inputs(spans, "traces")
    assert wf == "wf"
    assert (start_time, end_time) == (None, None)


def test_derive_query_inputs_missing_workflow_skips():
    spans = [_span({"span.type": "workflow"}, start=1, end=2)]
    with patch("monocle_test_tools.evals.okahu_eval.OkahuEval.get_fact_map",
               return_value={"traces": "trace_id"}):
        with pytest.raises(_DiscoverySkipped):
            _derive_query_inputs(spans, "traces")


# --- query body / time window ----------------------------------------------------

from monocle_test_tools.evals.okahu_eval_discovery import _query_fact_evals


class _FakeClient:
    api_base = "https://api.example/api"

    def __init__(self):
        self.captured = None

    def _paginate_post(self, url, body):
        self.captured = (url, body)
        return iter([])


def test_query_fact_evals_includes_time_window():
    client = _FakeClient()
    list(_query_fact_evals(client, "wf", ["0xabc123"], fact_name="traces",
                           start_time="2001-09-08T00:00:00.000Z",
                           end_time="2001-09-10T00:00:00.000Z"))
    url, body = client.captured
    assert url == "https://api.example/api/v1/workflows/wf/evals/query"
    assert body["fact_ids"] == ["abc123"]  # 0x stripped
    assert body["start_time"] == "2001-09-08T00:00:00.000Z"
    assert body["end_time"] == "2001-09-10T00:00:00.000Z"


def test_query_fact_evals_omits_window_when_absent():
    client = _FakeClient()
    list(_query_fact_evals(client, "wf", ["abc123"], fact_name="traces"))
    _url, body = client.captured
    assert "start_time" not in body and "end_time" not in body


# --- discover_fact_evals orchestration -------------------------------------------

from monocle_test_tools.evals.okahu_eval_discovery import discover_fact_evals


def _labeled_row(fact_id, label, eval_name=None, eval_id=None):
    return {"fact_id": fact_id, "eval_name": eval_name, "eval_id": eval_id,
            "eval_found": True, "eval_result": {"label": label, "explanation": "x"}}


def _traces_spans():
    return [_span({"span.type": "workflow", "workflow.name": "wf"},
                  trace_id=0xABC123, start=1_000_000_000_000_000_000, end=1_000_000_002_000_000_000)]


def _set_okahu_env(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "k")
    monkeypatch.setenv("OKAHU_EVALUATION_ENDPOINT", "https://eval.example/api")
    monkeypatch.setenv("OKAHU_API_ENDPOINT", "https://api.example")


def test_discover_returns_specs_and_classifies_custom(monkeypatch):
    _set_okahu_env(monkeypatch)
    # A builtin LLM eval, a builtin-slot eval, and a custom-template eval whose
    # name collides with a builtin ("hallucination"). Custom is keyed off eval_id.
    rows = [
        _labeled_row("abc123", "yes", eval_name="answer_relevancy",
                     eval_id="evaluation__traces__answer_relevancy"),
        _labeled_row("abc123", "major_hallucination", eval_name="hallucination",
                     eval_id="evaluation__traces__hallucination"),
        _labeled_row("abc123", "major_hallucination", eval_name="hallucination",
                     eval_id="custom_evaluation__generic__hallucination"),
        {"fact_id": "abc123", "eval_name": "bias",
         "eval_id": "evaluation__traces__bias", "eval_found": False},  # skipped
    ]
    with patch("monocle_test_tools.evals.okahu_eval_discovery._derive_query_inputs",
               return_value=("wf", ["abc123"], "2001-09-08T00:00:00.000Z", "2001-09-10T00:00:00.000Z")), \
         patch("monocle_test_tools.evals.okahu_eval_discovery._query_fact_evals",
               return_value=rows):
        specs, note = discover_fact_evals(_traces_spans(), fact_name="traces")

    assert note is None
    # 3 specs: answer_relevancy (builtin), hallucination (builtin), hallucination (custom).
    assert len(specs) == 3
    builtin = [s for s in specs if not s["_discovered_custom"]]
    custom = [s for s in specs if s["_discovered_custom"]]
    assert {s["criteria"] for s in builtin} == {"answer_relevancy", "hallucination"}
    for s in builtin:
        assert s["eval_type"] == "builtin"
        assert s["_discovered"] is True
        assert s["fact_name"] == "traces"
    assert len(custom) == 1
    assert custom[0]["criteria"] == "hallucination"
    assert custom[0]["eval_type"] == "custom"
    assert custom[0]["expected"] == "major_hallucination"


def test_discover_empty_when_no_labels(monkeypatch):
    _set_okahu_env(monkeypatch)
    with patch("monocle_test_tools.evals.okahu_eval_discovery._derive_query_inputs",
               return_value=("wf", ["abc123"], "2001-09-08T00:00:00.000Z", "2001-09-10T00:00:00.000Z")), \
         patch("monocle_test_tools.evals.okahu_eval_discovery._query_fact_evals",
               return_value=[]):
        specs, note = discover_fact_evals(_traces_spans(), fact_name="traces")
    assert specs == []
    assert note == "No existing evals found on this fact"


# --- BaseEval interface + OkahuEval override -------------------------------------

def test_base_eval_discovery_default_is_noop():
    """An evaluator that doesn't support discovery returns empty + a note, never raises."""
    from monocle_test_tools.evals.base_eval import BaseEval
    specs, note = BaseEval().discover_fact_evals(_traces_spans(), fact_name="traces")
    assert specs == []
    assert note and "not supported" in note


def test_okahu_eval_discovery_delegates_to_module():
    """OkahuEval.discover_fact_evals delegates to the okahu_eval_discovery module fn."""
    from monocle_test_tools.evals.okahu_eval import OkahuEval
    sentinel = ([{"criteria": "correctness"}], None)
    with patch("monocle_test_tools.evals.okahu_eval_discovery.discover_fact_evals",
               return_value=sentinel) as mod_fn:
        result = OkahuEval(eval_options={}).discover_fact_evals(_traces_spans(), fact_name="agentic_sessions")
    assert result == sentinel
    _args, kwargs = mod_fn.call_args
    assert kwargs["fact_name"] == "agentic_sessions"


def test_discover_missing_creds_skips(monkeypatch):
    monkeypatch.delenv("OKAHU_API_KEY", raising=False)
    specs, note = discover_fact_evals(_traces_spans(), fact_name="traces")
    assert specs == []
    assert note.startswith("eval discovery skipped:")


def test_discover_defaults_endpoints_to_prod(monkeypatch):
    # Only the API key is set; endpoints must default to prod, not skip discovery.
    monkeypatch.setenv("OKAHU_API_KEY", "k")
    monkeypatch.delenv("OKAHU_EVALUATION_ENDPOINT", raising=False)
    monkeypatch.delenv("OKAHU_API_ENDPOINT", raising=False)
    rows = [_labeled_row("abc123", "yes", eval_name="answer_relevancy",
                         eval_id="evaluation__traces__answer_relevancy")]
    with patch("monocle_test_tools.evals.okahu_eval_discovery._derive_query_inputs",
               return_value=("wf", ["abc123"], None, None)), \
         patch("monocle_test_tools.evals.okahu_eval_discovery._query_fact_evals",
               return_value=rows):
        specs, note = discover_fact_evals(_traces_spans(), fact_name="traces")
    assert note is None
    assert len(specs) == 1
    from monocle_test_tools.evals import okahu_filtered_eval as ofe
    client = ofe.OkahuFilteredEval.from_env()
    assert client.api_base == ofe.OKAHU_PROD_API_ENDPOINT + "/api"
    assert client.eval_base == ofe.OKAHU_PROD_EVALUATION_ENDPOINT.rstrip("/")


def test_discover_http_error_is_non_fatal(monkeypatch):
    _set_okahu_env(monkeypatch)
    with patch("monocle_test_tools.evals.okahu_eval_discovery._derive_query_inputs",
               side_effect=RuntimeError("boom")):
        specs, note = discover_fact_evals(_traces_spans(), fact_name="traces")
    assert specs == []
    assert note.startswith("eval discovery skipped:")
