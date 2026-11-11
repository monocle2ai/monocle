import json
from pathlib import Path

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.trace import SpanContext, TraceFlags

from monocle_test_tools.evals.monocle.template import EvidenceScope
from monocle_test_tools.evals.monocle.preprocess import build_evidence
from monocle_test_tools.file_span_loader import JSONSpanLoader

FIXTURES = Path(__file__).parent / "fixtures"
REFUND_TRACE = Path(__file__).parents[2] / "integration" / "json_loader_traces" / "642dbd9d0dfcfdbdc8849f67f34c8a19.json"
EVIDENCE_PATH = FIXTURES / "customer_support_refund.evidence.json"
TRACE_ID = "642dbd9d0dfcfdbdc8849f67f34c8a19"
THIN_PREAMBLE_OPENING = "You are evaluating an AI model's response."


def fixture_spans():
    fixture = json.loads(REFUND_TRACE.read_text())
    return JSONSpanLoader.from_json_str(json.dumps(fixture["spans"]))


def scope_with_span_types(span_types: list[str]) -> EvidenceScope:
    return EvidenceScope.default().model_copy(update={"span_types": span_types})


def span_types_in(evidence_text: str) -> set[str]:
    payload = json.loads(evidence_text)
    return {span["attributes"]["span.type"] for trace in payload["traces"] for span in trace["spans"]}


def test_fixture_trace_produces_identical_evidence():
    reference_prompt = json.loads(EVIDENCE_PATH.read_text())["prompt"]

    evidence = build_evidence(fixture_spans(), TRACE_ID, EvidenceScope.default())

    assert evidence.text == reference_prompt
    assert evidence.trace_id == TRACE_ID


def test_narrow_scope_drops_inference_spans():
    scope = scope_with_span_types(["agentic.tool.invocation"])

    evidence = build_evidence(fixture_spans(), TRACE_ID, scope)

    assert span_types_in(evidence.text) == {"agentic.tool.invocation"}


def test_inference_only_scope_uses_thin_preamble():
    scope = scope_with_span_types(["inference"])

    evidence = build_evidence(fixture_spans(), TRACE_ID, scope)

    assert json.loads(evidence.text)["instructions"].startswith(THIN_PREAMBLE_OPENING)
    assert span_types_in(evidence.text) == {"inference"}


def test_empty_after_filtering_returns_none():
    scope = scope_with_span_types(["retrieval"])

    assert build_evidence(fixture_spans(), TRACE_ID, scope) is None
