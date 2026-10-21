"""Unit tests for the AgentCore runner.

All tests use an injected fake boto3 client and a fake StreamingBody, so they
run without boto3, AWS credentials, or any network access.
"""
import json

import pytest

from monocle_test_tools.runner.runner import get_agent_runner, AgentTypes
from monocle_test_tools.runner.agentcore_runner import AgentCoreRunner

RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test_agent-AbCdEfGhIj"
ENDPOINT_ARN = RUNTIME_ARN + "/runtime-endpoint/DEFAULT"
# Same shape as Monocle's auto-generated session ids, which clear AgentCore's minimum.
VALID_SESSION_ID = "monocle_test_session_" + "a" * 32
# Workflow the deployed agent reports under — not the test's own workflow.
REMOTE_WORKFLOW = "deployed_agent_workflow"


class FakeStream:
    """Stand-in for a botocore StreamingBody."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def iter_lines(self, chunk_size=None):
        for line in self._body.split(b"\n"):
            yield line


class FakeClient:
    """Records the request and returns a canned AgentCore response."""

    def __init__(self, body: bytes = b'"Booked."', content_type: str = "application/json",
                 raise_error: Exception = None):
        self.body = body
        self.content_type = content_type
        self.raise_error = raise_error
        self.last_request = None
        self.call_count = 0

    def invoke_agent_runtime(self, **kwargs):
        self.call_count += 1
        self.last_request = kwargs
        if self.raise_error is not None:
            raise self.raise_error
        return {
            "response": FakeStream(self.body),
            "contentType": self.content_type,
            "statusCode": 200,
            "runtimeSessionId": kwargs.get("runtimeSessionId", "generated-by-aws"),
        }


def test_agent_type_mapping():
    assert isinstance(get_agent_runner(AgentTypes.AGENTCORE), AgentCoreRunner)
    assert AgentTypes.AGENTCORE == "agentcore"


def test_invokes_with_arn_payload_and_session():
    client = FakeClient()
    runner = AgentCoreRunner(client=client)

    result = runner.run_agent(RUNTIME_ARN, "Book a flight", session_id=VALID_SESSION_ID)

    request = client.last_request
    assert request["agentRuntimeArn"] == RUNTIME_ARN
    assert request["runtimeSessionId"] == VALID_SESSION_ID
    assert json.loads(request["payload"].decode("utf-8")) == {"prompt": "Book a flight"}
    assert request["contentType"] == "application/json"
    # A plain runtime ARN carries no qualifier, so AWS applies the default endpoint.
    assert "qualifier" not in request
    assert result == "Booked."


def test_endpoint_qualified_arn_is_split():
    client = FakeClient()
    runner = AgentCoreRunner(client=client)

    runner.run_agent(ENDPOINT_ARN, "hi", session_id=VALID_SESSION_ID)

    assert client.last_request["agentRuntimeArn"] == RUNTIME_ARN
    assert client.last_request["qualifier"] == "DEFAULT"


def test_explicit_qualifier_wins_over_arn():
    client = FakeClient()
    runner = AgentCoreRunner(client=client)

    runner.run_agent(ENDPOINT_ARN, "hi", session_id=VALID_SESSION_ID, qualifier="prod")

    assert client.last_request["qualifier"] == "prod"


def test_session_id_omitted_when_not_provided():
    client = FakeClient()
    runner = AgentCoreRunner(client=client)

    runner.run_agent(RUNTIME_ARN, "hi")

    # Omitting the parameter lets AgentCore generate the session id.
    assert "runtimeSessionId" not in client.last_request


def test_short_session_id_raises_without_padding():
    client = FakeClient()
    runner = AgentCoreRunner(client=client)

    with pytest.raises(ValueError) as excinfo:
        runner.run_agent(RUNTIME_ARN, "hi", session_id="monocle_test_session")

    message = str(excinfo.value)
    assert "33" in message
    # The id must be surfaced, not silently rewritten, and no call is made.
    assert "monocle_test_session" in message
    assert client.call_count == 0


def test_monocle_generated_session_id_is_accepted():
    client = FakeClient()
    runner = AgentCoreRunner(client=client)

    runner.run_agent(RUNTIME_ARN, "hi", session_id=VALID_SESSION_ID)

    assert len(VALID_SESSION_ID) >= 33
    assert client.last_request["runtimeSessionId"] == VALID_SESSION_ID


def test_dict_message_is_sent_verbatim():
    client = FakeClient()
    runner = AgentCoreRunner(client=client)

    runner.run_agent(RUNTIME_ARN, {"prompt": "hi", "locale": "en"}, session_id=VALID_SESSION_ID)

    assert json.loads(client.last_request["payload"].decode("utf-8")) == {
        "prompt": "hi", "locale": "en"
    }


def test_decodes_json_object_response():
    client = FakeClient(body=json.dumps({"response": "Booked."}).encode("utf-8"))
    runner = AgentCoreRunner(client=client)

    assert runner.run_agent(RUNTIME_ARN, "hi") == {"response": "Booked."}


def test_decodes_event_stream_response():
    body = b'data: Hello\ndata: world\n'
    client = FakeClient(body=body, content_type="text/event-stream")
    runner = AgentCoreRunner(client=client)

    assert runner.run_agent(RUNTIME_ARN, "hi") == "Hello\nworld"


def test_non_json_body_returned_as_text():
    client = FakeClient(body=b"plain text reply")
    runner = AgentCoreRunner(client=client)

    assert runner.run_agent(RUNTIME_ARN, "hi") == "plain text reply"


def test_aws_errors_propagate():
    class AccessDeniedException(Exception):
        pass

    client = FakeClient(raise_error=AccessDeniedException("not authorized"))
    runner = AgentCoreRunner(client=client)

    # The failure must surface rather than be returned as the agent's reply.
    with pytest.raises(AccessDeniedException):
        runner.run_agent(RUNTIME_ARN, "hi")


def test_rejects_non_arn_root_agent():
    runner = AgentCoreRunner(client=FakeClient())

    with pytest.raises(ValueError):
        runner.run_agent(None, "hi")


@pytest.mark.asyncio
async def test_run_agent_async_returns_decoded_response():
    client = FakeClient()
    runner = AgentCoreRunner(client=client)

    result = await runner.run_agent_async(RUNTIME_ARN, "hi", session_id=VALID_SESSION_ID)

    assert result == "Booked."
    assert client.last_request["runtimeSessionId"] == VALID_SESSION_ID


@pytest.mark.asyncio
async def test_run_agent_works_inside_a_running_loop():
    """run_agent falls back to a worker thread when a loop is already running."""
    client = FakeClient()
    runner = AgentCoreRunner(client=client)

    assert runner.run_agent(RUNTIME_ARN, "hi") == "Booked."


def test_injected_client_is_reused_across_calls():
    client = FakeClient()
    runner = AgentCoreRunner(client=client)

    runner.run_agent(RUNTIME_ARN, "one")
    runner.run_agent(RUNTIME_ARN, "two")

    assert client.call_count == 2
    assert runner._get_client() is client


def test_region_parsed_from_arn():
    assert AgentCoreRunner._region_from_arn(RUNTIME_ARN) == "us-west-2"
    assert AgentCoreRunner._region_from_arn(
        "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test_agent-AbCdEfGhIj"
    ) == "us-east-1"
    assert AgentCoreRunner._region_from_arn("not-an-arn") is None


def test_client_is_built_in_the_arn_region(monkeypatch):
    """The ARN's region wins over boto3's ambient default region.

    Without this, an agent deployed outside the caller's default region is
    reached at the wrong regional endpoint and fails as ResourceNotFoundException.
    """
    boto3 = pytest.importorskip("boto3")
    captured = {}

    def fake_client(service_name, **kwargs):
        captured["service_name"] = service_name
        captured["region_name"] = kwargs.get("region_name")
        return FakeClient()

    monkeypatch.setattr(boto3, "client", fake_client)

    AgentCoreRunner().run_agent(RUNTIME_ARN, "hi")

    assert captured["service_name"] == "bedrock-agentcore"
    assert captured["region_name"] == "us-west-2"


def test_explicit_region_overrides_arn_region(monkeypatch):
    boto3 = pytest.importorskip("boto3")
    captured = {}

    monkeypatch.setattr(
        boto3, "client",
        lambda service_name, **kw: captured.update(region_name=kw.get("region_name")) or FakeClient(),
    )

    AgentCoreRunner(region_name="eu-west-1").run_agent(RUNTIME_ARN, "hi")

    assert captured["region_name"] == "eu-west-1"


def test_remote_trace_hooks_are_inert():
    """The runner does not correlate the deployed agent's spans back to the test."""
    runner = AgentCoreRunner(client=FakeClient())

    assert runner.get_remote_traces_source() is None
    assert runner.get_remote_spans() == []


def test_no_remote_source_until_a_workflow_is_configured(monkeypatch):
    """Without a workflow name there is nothing to query, so retrieval is skipped."""
    monkeypatch.delenv("AGENTCORE_TRACE_WORKFLOW", raising=False)
    runner = AgentCoreRunner(client=FakeClient())

    runner.run_agent(RUNTIME_ARN, "hi", session_id=VALID_SESSION_ID)

    assert runner.get_remote_traces_source() is None
    assert runner.get_remote_trace_query() == {}


def test_remote_query_uses_the_session_id_that_was_sent():
    """The lookup key is the exact runtimeSessionId AgentCore received."""
    client = FakeClient()
    runner = AgentCoreRunner(client=client, trace_workflow_name=REMOTE_WORKFLOW)

    runner.run_agent(RUNTIME_ARN, "hi", session_id=VALID_SESSION_ID)

    assert client.last_request["runtimeSessionId"] == VALID_SESSION_ID
    assert runner.get_remote_traces_source() == "okahu"
    assert runner.get_remote_trace_query() == {
        "id": VALID_SESSION_ID,
        "fact_name": "session",
        "workflow_name": REMOTE_WORKFLOW,
    }


def test_remote_query_tracks_the_latest_session():
    """Each invocation retargets the lookup at that call's session."""
    runner = AgentCoreRunner(client=FakeClient(), trace_workflow_name=REMOTE_WORKFLOW)
    second_session = "monocle_test_session_" + "b" * 32

    runner.run_agent(RUNTIME_ARN, "one", session_id=VALID_SESSION_ID)
    runner.run_agent(RUNTIME_ARN, "two", session_id=second_session)

    assert runner.get_remote_trace_query()["id"] == second_session


def test_workflow_name_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("AGENTCORE_TRACE_WORKFLOW", REMOTE_WORKFLOW)
    runner = AgentCoreRunner(client=FakeClient())

    runner.run_agent(RUNTIME_ARN, "hi", session_id=VALID_SESSION_ID)

    assert runner.get_remote_trace_query()["workflow_name"] == REMOTE_WORKFLOW


def test_aws_generated_session_is_not_used_for_lookup():
    """AgentCore generates a session when none is sent, but it is not on the request.

    Correlation would need the id the agent actually stamped on its spans, so
    the runner reports no query rather than guessing one.
    """
    runner = AgentCoreRunner(client=FakeClient(), trace_workflow_name=REMOTE_WORKFLOW)

    runner.run_agent(RUNTIME_ARN, "hi")

    assert runner.get_remote_trace_query() == {}
    assert runner.get_remote_traces_source() is None


if __name__ == "__main__":
    pytest.main([__file__])
