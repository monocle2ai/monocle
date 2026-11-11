import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from monocle_apptrace.providers.okahu.okahu_eval_result_exporter import OkahuEvalResultExporter
from monocle_test_tools.providers.okahu.okahu_eval import OkahuEval

MONOCLE_TEMPLATE_PATH = Path(__file__).parents[1] / "integration" / "custom_templates" / "refund_confirmation_grounding.json"
PLAIN_TEMPLATE = {
    "name": "plain",
    "eval_prompt": "judge it",
    "structure_output": {"label": {"enums": ["a", "b"], "description": "x"}},
}


def _judge_response():
    response = MagicMock()
    response.headers = {"Content-Type": "application/json"}
    response.raise_for_status.return_value = None
    response.json.return_value = {"job_id": "job_1", "result": [{"result": json.dumps({"label": "a", "explanation": "ok"})}]}
    return response


def _request_body_sent_to_okahu(monkeypatch, template):
    span = MagicMock()
    span.attributes = {"workflow.name": "wf"}
    monkeypatch.setenv("OKAHU_API_KEY", "key")
    with patch.object(OkahuEval, "export_trace", return_value="traceid"), \
         patch.object(OkahuEval, "enumerate_fact_ids", return_value=["traceid"]), \
         patch("monocle_test_tools.providers.okahu.okahu_eval.requests.post", return_value=_judge_response()) as post, \
         patch.object(OkahuEvalResultExporter, "export_results", return_value=None):
        OkahuEval(eval_options={"trace_source": "file"}).evaluate(
            filtered_spans=[span], template=template, fact_name="traces")
    return post.call_args.kwargs["json"]


def test_result_export_sends_the_template_okahu_accepts(monkeypatch):
    monocle_template = json.loads(MONOCLE_TEMPLATE_PATH.read_text(encoding="utf-8"))
    span = MagicMock()
    span.attributes = {"workflow.name": "wf"}
    monkeypatch.setenv("OKAHU_API_KEY", "key")
    monkeypatch.setenv("MONOCLE_EXPORTER", "okahu")
    with patch.object(OkahuEval, "export_trace", return_value="traceid"), \
         patch.object(OkahuEval, "enumerate_fact_ids", return_value=["traceid"]), \
         patch("monocle_test_tools.providers.okahu.okahu_eval.requests.post", return_value=_judge_response()), \
         patch.object(OkahuEvalResultExporter, "export_results", return_value=None) as export_results:
        OkahuEval(eval_options={"trace_source": "file"}).evaluate(
            filtered_spans=[span], template=monocle_template, fact_name="traces")

    exported = export_results.call_args.kwargs["template"]
    assert "llm" not in exported
    assert "evidence_scope" not in exported
    assert exported["name"] == "refund_confirmation_grounding"


def test_okahu_request_lifts_evidence_scope_and_drops_model(monkeypatch):
    monocle_template = json.loads(MONOCLE_TEMPLATE_PATH.read_text(encoding="utf-8"))

    body = _request_body_sent_to_okahu(monkeypatch, monocle_template)

    assert body["evidence_scope"] == monocle_template["evidence_scope"]
    assert "evidence_scope" not in body["template"]
    assert "llm" not in body["template"]
    assert body["template"]["name"] == "refund_confirmation_grounding"


def test_template_with_profile_keeps_llm_block(monkeypatch):
    template = {**PLAIN_TEMPLATE, "llm": {"profile": "fast", "model": "openai:gpt-4.1-mini", "min_confidence": 0.8}}

    body = _request_body_sent_to_okahu(monkeypatch, template)

    assert body["template"]["llm"] == {"profile": "fast"}
    assert "evidence_scope" not in body


def test_template_without_extras_is_sent_unchanged(monkeypatch):
    template = copy.deepcopy(PLAIN_TEMPLATE)

    body = _request_body_sent_to_okahu(monkeypatch, template)

    assert body == {"template": PLAIN_TEMPLATE}
    assert template == PLAIN_TEMPLATE
