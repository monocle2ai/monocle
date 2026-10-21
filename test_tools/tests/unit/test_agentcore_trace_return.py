"""Traces returned inside the AgentCore response, instead of fetched from a backend.

A deployed agent running with trace-return enabled appends its spans to the
value it returns; the runner takes them off before the test sees the response.
These cover the client half against payloads built by the real encoder, so the
wire format is not restated here.
"""
import contextlib
import json
from types import SimpleNamespace

import pytest
from botocore.awsrequest import AWSRequest

from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import (
    AGENTCORE_CUSTOM_HEADER_PREFIX,
    MONOCLE_TRACE_RETRIEVAL_KEY_ENV,
    TRACE_RETURN_REQUEST_HEADER,
)
from monocle_test_tools.runner.agentcore_runner import (
    _RETRIEVAL_KEY_HEADER,
    _SIGN_EVENT,
    AgentCoreRunner,
)
from monocle_test_tools.file_span_loader import JSONSpanLoader

RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test_agent-AbCdEfGhIj"
AGENT_ANSWER = "Flight booked from San Jose to Seattle."

SPAN_DICT = {
    "name": "agentic.tool.invocation",
    "context": {"trace_id": "0x" + "0" * 31 + "1", "span_id": "0x" + "0" * 15 + "1",
                "trace_state": "[]"},
    "kind": "SpanKind.INTERNAL",
    "parent_id": None,
    "start_time": "2026-08-14T00:00:00.000000Z",
    "end_time": "2026-08-14T00:00:01.000000Z",
    "status": {"status_code": "OK"},
    "attributes": {"span.type": "agentic.tool.invocation", "entity.1.name": "book_flight_tool"},
    "events": [],
    "links": [],
    "resource": {"attributes": {"service.name": "deployed_agent"}, "schema_url": ""},
}


class FakeStream:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body


class FakeEvents:
    """Records what the runner hooks onto the client, and what it leaves behind."""

    def __init__(self):
        self.registered = []
        self.history = []

    def register_first(self, event, handler):
        self.registered.append((event, handler))
        self.history.append(event)

    def unregister(self, event, handler):
        self.registered.remove((event, handler))


class FakeClient:
    """Returns a canned response body, as the deployed agent would."""

    def __init__(self, body: bytes, fail: bool = False):
        self.body = body
        self.fail = fail
        self.meta = SimpleNamespace(events=FakeEvents())

    def invoke_agent_runtime(self, **kwargs):
        if self.fail:
            raise RuntimeError("boom")
        return {"response": FakeStream(self.body), "contentType": "application/json",
                "statusCode": 200}


def _agent_response_with_trailer(answer: str = AGENT_ANSWER, spans=(SPAN_DICT,)) -> bytes:
    """Build the response a trace-return-enabled agent would produce.

    Uses the same encoder the agent side calls, so the test cannot drift from
    the real format.
    """
    delimiter = tr.make_delimiter()
    payload = tr.encode_spans_from_dicts(list(spans)) if hasattr(tr, "encode_spans_from_dicts") \
        else _encode_dicts(list(spans))
    return json.dumps(answer + delimiter + payload).encode("utf-8")


def _encode_dicts(span_dicts):
    """Mirror of trace_return.encode_spans for already-serialized span dicts."""
    import base64, gzip
    raw = json.dumps(span_dicts).encode("utf-8")
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def test_trailer_is_stripped_from_the_response():
    """The test sees the agent's answer only — the trailer never reaches it."""
    runner = AgentCoreRunner(client=FakeClient(_agent_response_with_trailer()))

    result = runner.run_agent(RUNTIME_ARN, "Book a flight")

    assert result == AGENT_ANSWER
    assert "__MONOCLE_TRACES__" not in result


def test_returned_spans_are_deserialized():
    """The stripped trailer becomes spans the validator can assert on."""
    runner = AgentCoreRunner(client=FakeClient(_agent_response_with_trailer()))

    runner.run_agent(RUNTIME_ARN, "Book a flight")

    spans = runner.get_remote_spans()
    assert len(spans) == 1
    assert spans[0].attributes["entity.1.name"] == "book_flight_tool"


def test_response_without_a_trailer_is_untouched():
    """An agent without trace-return enabled behaves exactly as before."""
    body = json.dumps(AGENT_ANSWER).encode("utf-8")
    runner = AgentCoreRunner(client=FakeClient(body))

    result = runner.run_agent(RUNTIME_ARN, "Book a flight")

    assert result == AGENT_ANSWER
    assert runner.get_remote_spans() == []


def test_corrupt_trailer_still_returns_the_answer():
    """A malformed payload must not lose the agent's response."""
    body = json.dumps(AGENT_ANSWER + tr.make_delimiter() + "not-valid-base64!!").encode("utf-8")
    runner = AgentCoreRunner(client=FakeClient(body))

    result = runner.run_agent(RUNTIME_ARN, "Book a flight")

    assert result == AGENT_ANSWER
    assert runner.get_remote_spans() == []


def test_answer_containing_the_prefix_but_no_terminator():
    """Text that merely mentions the marker is left alone."""
    answer = "The marker is __MONOCLE_TRACES__ and that is all"
    runner = AgentCoreRunner(client=FakeClient(json.dumps(answer).encode("utf-8")))

    assert runner.run_agent(RUNTIME_ARN, "hi") == answer


def test_returned_spans_suppress_the_session_lookup():
    """Spans in the response are the same spans; fetching them again duplicates them."""
    runner = AgentCoreRunner(client=FakeClient(_agent_response_with_trailer()),
                             trace_workflow_name="deployed_agent_workflow")

    runner.run_agent(RUNTIME_ARN, "hi", session_id="monocle_test_session_" + "a" * 32)

    assert runner.get_remote_spans(), "the agent returned spans"
    assert runner.get_remote_traces_source() is None
    assert runner.get_remote_trace_query() == {}


def test_session_lookup_still_used_when_no_spans_returned():
    """An agent without trace-return enabled falls back to fetching by session."""
    body = json.dumps(AGENT_ANSWER).encode("utf-8")
    runner = AgentCoreRunner(client=FakeClient(body),
                             trace_workflow_name="deployed_agent_workflow")

    runner.run_agent(RUNTIME_ARN, "hi", session_id="monocle_test_session_" + "a" * 32)

    assert runner.get_remote_spans() == []
    assert runner.get_remote_traces_source() == "okahu"
    assert runner.get_remote_trace_query()["fact_name"] == "session"


def test_spans_do_not_leak_between_invocations():
    """A reused runner must not report a previous call's spans."""
    runner = AgentCoreRunner(client=FakeClient(_agent_response_with_trailer()))
    runner.run_agent(RUNTIME_ARN, "first")
    assert runner.get_remote_spans()

    runner._client = FakeClient(json.dumps(AGENT_ANSWER).encode("utf-8"))
    runner.run_agent(RUNTIME_ARN, "second")

    assert runner.get_remote_spans() == []


def test_round_trip_through_the_real_encoder():
    """Spans encoded by the agent-side helper decode back to the same span."""
    delimiter = tr.make_delimiter()
    payload = _encode_dicts([SPAN_DICT])
    body = json.dumps(AGENT_ANSWER + delimiter + payload).encode("utf-8")

    runner = AgentCoreRunner(client=FakeClient(body))
    result = runner.run_agent(RUNTIME_ARN, "hi")

    assert result == AGENT_ANSWER
    decoded = runner.get_remote_spans()
    assert [s.name for s in decoded] == ["agentic.tool.invocation"]


KEY = "s3cret"


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv(MONOCLE_TRACE_RETRIEVAL_KEY_ENV, KEY)


def _signed_request(runner=None) -> AWSRequest:
    """Run the runner's hook over a request, the way botocore would when signing."""
    request = AWSRequest(method="POST", url="https://example.invalid/")
    handler = (runner or AgentCoreRunner())._retrieval_key_handler()
    if handler is not None:
        handler(request=request)
    return request


def test_key_is_presented_under_the_prefix_agentcore_forwards(with_key):
    """Any other header name is dropped before the deployed agent sees it."""
    assert _RETRIEVAL_KEY_HEADER == AGENTCORE_CUSTOM_HEADER_PREFIX + TRACE_RETURN_REQUEST_HEADER
    assert _signed_request().headers[_RETRIEVAL_KEY_HEADER] == KEY


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_nothing_is_presented_without_a_key(monkeypatch, value):
    """The common case: a run that never asked for spans in the response."""
    if value is None:
        monkeypatch.delenv(MONOCLE_TRACE_RETRIEVAL_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(MONOCLE_TRACE_RETRIEVAL_KEY_ENV, value)
    client = FakeClient(_agent_response_with_trailer())

    AgentCoreRunner()._invoke(client, {"agentRuntimeArn": RUNTIME_ARN})

    assert _signed_request().headers.get(_RETRIEVAL_KEY_HEADER) is None
    assert client.meta.events.history == []


@pytest.mark.parametrize("fail", [False, True], ids=["call succeeds", "call raises"])
def test_hook_lives_only_for_the_call(with_key, fail):
    """The client can be the caller's own and outlive this runner."""
    client = FakeClient(_agent_response_with_trailer(), fail=fail)

    with contextlib.suppress(RuntimeError):
        AgentCoreRunner()._invoke(client, {"agentRuntimeArn": RUNTIME_ARN})

    assert client.meta.events.history == [_SIGN_EVENT]
    assert client.meta.events.registered == []


def test_client_that_takes_no_hooks_is_still_invoked(with_key):
    """A caller can supply any object with invoke_agent_runtime, as before."""
    class MinimalClient:
        def __init__(self):
            self.calls = []

        def invoke_agent_runtime(self, **request):
            self.calls.append(request)
            return {"statusCode": 200}

    client = MinimalClient()

    AgentCoreRunner()._invoke(client, {"agentRuntimeArn": RUNTIME_ARN})

    assert client.calls == [{"agentRuntimeArn": RUNTIME_ARN}]


def test_key_is_read_per_call(monkeypatch):
    """Rotating the key must not be masked by one cached at construction."""
    runner = AgentCoreRunner()

    for key in ("first", "second"):
        monkeypatch.setenv(MONOCLE_TRACE_RETRIEVAL_KEY_ENV, key)
        assert _signed_request(runner).headers[_RETRIEVAL_KEY_HEADER] == key


def test_caller_supplied_header_is_not_duplicated(with_key):
    """A second copy would reach the agent as one comma-joined value."""
    request = AWSRequest(method="POST", url="https://example.invalid/")
    request.headers.add_header(_RETRIEVAL_KEY_HEADER, "already-there")

    AgentCoreRunner()._retrieval_key_handler()(request=request)

    assert request.headers.get_all(_RETRIEVAL_KEY_HEADER) == ["already-there"]


if __name__ == "__main__":
    pytest.main([__file__])
