import json
import pytest
azure_functions = pytest.importorskip("azure.functions")
from monocle_apptrace.instrumentation.metamodel.azfunc._helper import azureSpanHandler
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
    def to_json(self): return '{"name": "azfunc_child"}'


def test_azfunc_injects_trailer(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(21)])
    resp = azure_functions.HttpResponse(body=json.dumps({"answer": "hi"}).encode("utf-8"),
                                        status_code=200, mimetype="application/json")
    azureSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=resp, ex=None, span=FakeSpan(21), parent_span=None)
    hv = resp.headers.get("x-monocle-traces")
    assert hv is not None
    delim = tr.parse_delimiter_from_header(hv)
    body = resp.get_body()
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert json.loads(clean.decode())["answer"] == "hi"
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "azfunc_child"


def test_azfunc_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    get_trace_return_exporter().clear()
    resp = azure_functions.HttpResponse(body=b"{}", status_code=200)
    azureSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=resp, ex=None, span=FakeSpan(21), parent_span=None)
    assert resp.get_body() == b"{}"
    assert resp.headers.get("x-monocle-traces") is None


def test_azfunc_no_header_when_body_mutation_fails(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(22)])
    resp = azure_functions.HttpResponse(body=b"{}", status_code=200)
    # Freeze get_body so the post-setattr verification sees no change,
    # simulating a future private-attr rename.
    monkeypatch.setattr(resp, "get_body", lambda: b"{}")
    azureSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=resp, ex=None, span=FakeSpan(22), parent_span=None)
    assert resp.headers.get("x-monocle-traces") is None
