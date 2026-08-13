"""Live integration test against an agent deployed to AWS Bedrock AgentCore Runtime.

Requires a deployed AgentCore agent and AWS credentials, so it is skipped unless
``AGENTCORE_RUNTIME_URL`` is set. Both the plain runtime ARN and the
endpoint-qualified form are accepted:

    export AGENTCORE_RUNTIME_URL=arn:aws:bedrock-agentcore:<region>:<account>:runtime/<agent-id>

The AWS region is taken from the ARN, so no region variable is needed.

The span test additionally needs ``AGENTCORE_TRACE_WORKFLOW`` (the Okahu
workflow the deployed agent exports under) and an ``OKAHU_API_KEY`` for the
tenant receiving those traces.
"""
import os
import time
import uuid

import pytest
import requests
from dotenv import load_dotenv

from monocle_test_tools import TraceAssertion

load_dotenv()

AGENTCORE_RUNTIME_URL = os.getenv("AGENTCORE_RUNTIME_URL")
AGENTCORE_TRACE_WORKFLOW = os.getenv("AGENTCORE_TRACE_WORKFLOW")

# Spans are exported from inside AWS, so they land in Okahu a little after the
# call returns.
TRACE_WAIT_SECONDS = 120
TRACE_POLL_SECONDS = 5

pytestmark = pytest.mark.skipif(
    not AGENTCORE_RUNTIME_URL,
    reason="AGENTCORE_RUNTIME_URL is not set; requires a deployed AgentCore agent and AWS credentials.",
)


def _session_id() -> str:
    return f"monocle_test_session_{uuid.uuid4().hex}"


def _load_session_spans(asserter: TraceAssertion, session_id: str) -> None:
    """Load the deployed agent's spans for a session, waiting for them to arrive.

    The agent stamps the ``runtimeSessionId`` it was invoked with onto its spans
    as ``scope.agentic.session``, which Okahu indexes as the ``agent_sessions``
    fact — so the session id generated here is enough to find them.
    """
    deadline = time.monotonic() + TRACE_WAIT_SECONDS
    last_error = None
    while time.monotonic() < deadline:
        try:
            asserter.with_trace_source(
                "okahu", id=session_id, fact_name="session",
                workflow_name=AGENTCORE_TRACE_WORKFLOW,
            )
            return
        except requests.HTTPError as exc:
            # 4xx other than "not found yet" means the query itself is rejected
            # (e.g. the tenant has no agent_sessions fact) — retrying cannot help.
            status = getattr(exc.response, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 404:
                raise AssertionError(
                    f"Okahu rejected the session lookup for workflow "
                    f"'{AGENTCORE_TRACE_WORKFLOW}' (HTTP {status}): "
                    f"{getattr(exc.response, 'text', '')[:200]}"
                ) from exc
            last_error = exc
            time.sleep(TRACE_POLL_SECONDS)
        except ConnectionError as exc:
            last_error = exc
            time.sleep(TRACE_POLL_SECONDS)
    raise AssertionError(
        f"No spans for session '{session_id}' appeared in Okahu workflow "
        f"'{AGENTCORE_TRACE_WORKFLOW}' within {TRACE_WAIT_SECONDS}s: {last_error}"
    )


def test_agentcore_remote_agent_response(monocle_trace_asserter: TraceAssertion):
    """End-to-end: prompt in, the deployed agent's text answer out."""
    response = monocle_trace_asserter.run_agent(
        AGENTCORE_RUNTIME_URL,
        "agentcore",
        "Book a flight from San Jose to Seattle for 22 oct 2026",
        session_id=_session_id(),
    )

    assert isinstance(response, str), f"expected a str response, got {type(response).__name__}"
    assert "San Jose" in response
    assert "Seattle" in response


def test_agentcore_remote_agent_rejects_short_session_id(monocle_trace_asserter: TraceAssertion):
    """A session id shorter than AgentCore's minimum fails fast and clearly."""
    with pytest.raises(ValueError, match="33"):
        monocle_trace_asserter.run_agent(
            AGENTCORE_RUNTIME_URL,
            "agentcore",
            "Book a flight from San Jose to Seattle for 22 Nov 2026",
            session_id="short_session",
        )


@pytest.mark.skipif(
    not AGENTCORE_TRACE_WORKFLOW,
    reason="AGENTCORE_TRACE_WORKFLOW is not set; requires an Okahu API key for the tenant "
           "the deployed agent exports to.",
)
def test_agentcore_remote_agent_response_and_spans(monocle_trace_asserter: TraceAssertion):
    """Response plus the spans the deployed agent produced for that same call.

    The runner itself emits no spans — everything asserted here was produced
    inside the deployed agent and retrieved from Okahu by session id.
    """
    session_id = _session_id()

    response = monocle_trace_asserter.run_agent(
        AGENTCORE_RUNTIME_URL,
        "agentcore",
        "Book a flight from San Jose to Seattle for 22 Oct 2026",
        session_id=session_id,
    )

    assert isinstance(response, str), f"expected a str response, got {type(response).__name__}"
    assert "Seattle" in response

    _load_session_spans(monocle_trace_asserter, session_id)

    monocle_trace_asserter.called_agent("agc_travel_agent")
    monocle_trace_asserter.called_tool("book_flight_tool")
    monocle_trace_asserter.has_scope("agentic.session", session_id)


@pytest.mark.skipif(
    not AGENTCORE_TRACE_WORKFLOW,
    reason="AGENTCORE_TRACE_WORKFLOW is not set; requires an Okahu API key for the tenant "
           "the deployed agent exports to.",
)
def test_agentcore_remote_spans_are_retrieved_by_session(monocle_trace_asserter: TraceAssertion):
    """The same spans, retrieved without asking for them explicitly.

    ``run_agent`` reports the session as the runner's remote trace query, so the
    spans the agent produced inside AWS are imported and asserted on like local
    ones — no ``with_trace_source`` call, and no waiting loop in the test.
    """
    session_id = _session_id()

    response = monocle_trace_asserter.run_agent(
        AGENTCORE_RUNTIME_URL,
        "agentcore",
        "Book a flight from San Jose to Seattle for 22 Oct 2026",
        session_id=session_id,
    )

    assert isinstance(response, str)
    assert "Seattle" in response

    monocle_trace_asserter.called_agent("agc_travel_agent")
    monocle_trace_asserter.called_tool("book_flight_tool")
    monocle_trace_asserter.has_scope("agentic.session", session_id)


if __name__ == "__main__":
    pytest.main([__file__])
