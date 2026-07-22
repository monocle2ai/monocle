from monocle_apptrace.instrumentation.metamodel.requests._helper import RequestSpanHandler
from monocle_apptrace.instrumentation.common import trace_return as tr


class FakeResponse:
    def __init__(self, content, headers):
        self._content = content
        self.headers = headers


def test_strip_and_stash():
    delim = tr.make_delimiter()

    class FakeSpan:
        def to_json(self): return '{"name": "inference"}'
    trailer = tr.build_trailer_bytes([FakeSpan()], delim)
    resp = FakeResponse(b'{"answer": "hi"}' + trailer,
                        {"x-monocle-traces": tr.build_response_header_value(delim)})
    RequestSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=resp, ex=None, span=None, parent_span=None)
    assert resp._content == b'{"answer": "hi"}'
    import json
    spans = json.loads(resp._monocle_remote_spans)
    assert spans[0]["name"] == "inference"


def test_no_header_is_noop():
    resp = FakeResponse(b'{"answer": "hi"}', {})
    RequestSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=resp, ex=None, span=None, parent_span=None)
    assert resp._content == b'{"answer": "hi"}'
    assert not hasattr(resp, "_monocle_remote_spans")
