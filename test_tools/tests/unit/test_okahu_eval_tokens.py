import json
from unittest.mock import patch, MagicMock
from monocle_test_tools.evals.okahu_eval import OkahuEval

def _resp(judge: dict):
    m = MagicMock()
    m.headers = {"Content-Type": "application/json"}
    m.raise_for_status.return_value = None
    m.json.return_value = {"job_id": "interactive_x_1", "result": [{"result": json.dumps(judge)}]}
    return m

def test_evaluate_stashes_total_tokens_and_output(monkeypatch):
    ev = OkahuEval(eval_options={"trace_source": "file"})
    judge = {"label": "major_hallucination", "explanation": "why", "total_tokens": 512}
    span = MagicMock()
    span.attributes = {"workflow.name": "wf"}
    monkeypatch.setenv("OKAHU_API_KEY", "k")
    with patch.object(OkahuEval, "export_trace", return_value="traceid"), \
         patch.object(OkahuEval, "enumerate_fact_ids", return_value=["traceid"]), \
         patch("monocle_test_tools.evals.okahu_eval.requests.post", return_value=_resp(judge)):
        label, explanation = ev.evaluate(filtered_spans=[span], template={"name": "t"}, fact_name="traces")
    assert (label, explanation) == ("major_hallucination", "why")   # arity unchanged
    assert ev.last_total_tokens == 512
    assert ev.last_judge_output["label"] == "major_hallucination"
