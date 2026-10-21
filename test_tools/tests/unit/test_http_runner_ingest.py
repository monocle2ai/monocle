import json
from monocle_test_tools.runner.runner import get_agent_runner, AgentTypes
from monocle_test_tools.runner.http_runner import HttpRunner, HttpOkahuRunner


def test_agent_type_mapping():
    assert isinstance(get_agent_runner(AgentTypes.HTTP), HttpRunner)
    assert isinstance(get_agent_runner(AgentTypes.HTTP_WITH_OKAHU), HttpOkahuRunner)
    assert AgentTypes.HTTP == "http"
    assert AgentTypes.HTTP_WITH_OKAHU == "http_with_okahu"


def test_httprunner_extracts_stashed_spans():
    class FakeResponse:
        pass
    resp = FakeResponse()
    # minimal file-format span dict
    resp._monocle_remote_spans = json.dumps([{
        "name": "inference",
        "context": {"trace_id": "0x" + "0"*31 + "1", "span_id": "0x" + "0"*15 + "1", "trace_state": "[]"},
        "kind": "SpanKind.INTERNAL",
        "parent_id": None,
        "start_time": "2026-07-21T00:00:00.000000Z",
        "end_time": "2026-07-21T00:00:01.000000Z",
        "status": {"status_code": "OK"},
        "attributes": {"span.type": "inference"},
        "events": [],
        "links": [],
        "resource": {"attributes": {"service.name": "test"}, "schema_url": ""}
    }])
    runner = HttpRunner()
    runner._capture_remote_spans(resp)
    spans = runner.get_remote_spans()
    assert len(spans) == 1
    assert spans[0].name == "inference"
    assert runner.get_remote_traces_source() is None
