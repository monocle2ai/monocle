import json
from monocle_apptrace.instrumentation.metamodel.lambdafunc._helper import lambdaSpanHandler
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.exporters.trace_return_exporter import get_trace_return_exporter


class FakeCtx:
    def __init__(self, tid): self.trace_id = tid
class FakeSpan:
    def __init__(self, tid):
        self._a = {"monocle_apptrace.version": "1.0", "scope.monocle_trace_return": "true"}
        self._c = FakeCtx(tid)
    @property
    def attributes(self): return self._a
    def get_span_context(self): return self._c
    def to_json(self): return '{"name": "lambda_child"}'


def test_lambda_injects_trailer(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(11)])
    result = {"statusCode": 200, "headers": {"Content-Type": "application/json"},
              "body": json.dumps({"answer": "hi"})}
    lambdaSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=result, ex=None, span=FakeSpan(11), parent_span=None)
    # header set
    assert any(k.lower() == "x-monocle-traces" for k in result["headers"])
    hv = next(v for k, v in result["headers"].items() if k.lower() == "x-monocle-traces")
    delim = tr.parse_delimiter_from_header(hv)
    clean, payload = tr.split_body_and_trailer(result["body"].encode("utf-8"), delim)
    assert json.loads(clean.decode())["answer"] == "hi"
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "lambda_child"


def test_lambda_skips_base64_body(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(11)])
    result = {"statusCode": 200, "headers": {}, "body": "aGVsbG8=", "isBase64Encoded": True}
    lambdaSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=result, ex=None, span=FakeSpan(11), parent_span=None)
    assert result["body"] == "aGVsbG8="  # untouched
    assert all(k.lower() != "x-monocle-traces" for k in result["headers"])


def test_lambda_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    get_trace_return_exporter().clear()
    result = {"statusCode": 200, "headers": {}, "body": "{}"}
    lambdaSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=result, ex=None, span=FakeSpan(11), parent_span=None)
    assert result == {"statusCode": 200, "headers": {}, "body": "{}"}
