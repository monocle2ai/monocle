"""Live test against an agent deployed to Vertex AI Agent Engine.

The deployed agent is apptrace/tests/integration/servers/vertex_agent_engine/monocle_adk_agent,
deployed with ``adk deploy agent_engine --otel_to_cloud``.

Invokes the deployed agent and reads its spans back from Okahu. Needs
AGENT_ENGINE_RESOURCE_NAME, AGENT_ENGINE_TRACE_WORKFLOW (registered in an Okahu
app), OKAHU_API_KEY (plus OKAHU_API_ENDPOINT outside prod) and Google
application-default credentials.
"""
import asyncio
import os
import re
import time
import uuid

import pytest
import requests

from monocle_test_tools import TraceAssertion

AGENT_ENGINE_RESOURCE_NAME = os.getenv("AGENT_ENGINE_RESOURCE_NAME")
AGENT_ENGINE_TRACE_WORKFLOW = os.getenv("AGENT_ENGINE_TRACE_WORKFLOW")

# Spans reach Okahu a little after the call returns.
TRACE_WAIT_SECONDS = 120
TRACE_POLL_SECONDS = 5

_RESOURCE_RE = re.compile(r"^projects/([^/]+)/locations/([^/]+)/reasoningEngines/([^/]+)$")

pytestmark = pytest.mark.skipif(
    not (AGENT_ENGINE_RESOURCE_NAME and AGENT_ENGINE_TRACE_WORKFLOW),
    reason="AGENT_ENGINE_RESOURCE_NAME or AGENT_ENGINE_TRACE_WORKFLOW is not set; "
           "requires a deployed Agent Engine agent, Google credentials and an Okahu API key.",
)


def _query_agent_engine(message: str) -> str:
    """Run one turn and return the session the agent created for it.

    Not ``async_create_session``: Agent Engine does not sample queries into a
    pre-created session, so they leave no trace.
    """
    import vertexai

    project, location, _ = _RESOURCE_RE.match(AGENT_ENGINE_RESOURCE_NAME).groups()
    agent = vertexai.Client(project=project, location=location).agent_engines.get(
        name=AGENT_ENGINE_RESOURCE_NAME)
    user_id = f"monocle_test_user_{uuid.uuid4().hex}"

    async def run() -> str:
        events = [event async for event in agent.async_stream_query(message=message, user_id=user_id)]
        errors = [e.get("error_message") for e in events if e.get("error_code")]
        assert events and not errors, f"the deployed agent failed: {errors or 'no events'}"
        sessions = (await agent.async_list_sessions(user_id=user_id))["sessions"]
        assert len(sessions) == 1, f"expected one session for {user_id}, got {len(sessions)}"
        return sessions[0]["id"]

    return asyncio.run(run())


def _load_session_spans(asserter: TraceAssertion, session_id: str) -> None:
    """Load the session's spans from Okahu, waiting for them to arrive."""
    deadline = time.monotonic() + TRACE_WAIT_SECONDS
    last_error = None
    while time.monotonic() < deadline:
        try:
            asserter.with_trace_source(
                "okahu", id=session_id, fact_name="session",
                workflow_name=AGENT_ENGINE_TRACE_WORKFLOW,
            )
            return
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 404:
                raise AssertionError(
                    f"Okahu rejected the session lookup for workflow "
                    f"'{AGENT_ENGINE_TRACE_WORKFLOW}' (HTTP {status}): "
                    f"{getattr(exc.response, 'text', '')[:200]}"
                ) from exc
            last_error = exc
        except ConnectionError as exc:
            last_error = exc
        time.sleep(TRACE_POLL_SECONDS)
    raise AssertionError(
        f"No spans for session '{session_id}' appeared in Okahu workflow "
        f"'{AGENT_ENGINE_TRACE_WORKFLOW}' within {TRACE_WAIT_SECONDS}s: {last_error}"
    )


def test_agent_engine_remote_agent_spans(monocle_trace_asserter: TraceAssertion):
    """The deployed agent's own spans, read back from Okahu for the session it ran in."""
    session_id = _query_agent_engine("Book a flight from BOM to JFK.")

    _load_session_spans(monocle_trace_asserter, session_id)

    # Existing ADK and Gemini instrumentation.
    monocle_trace_asserter.where(attribute={"span.type": "agentic.turn"})
    monocle_trace_asserter.called_agent("flight_booking_agent")
    monocle_trace_asserter.called_tool("book_flight", agent_name="flight_booking_agent") \
        .contains_output("FLIGHT CONFIRMED")
    monocle_trace_asserter.where(
        attribute={"span.type": "inference", "entity.1.type": "inference.vertexai",
                   "entity.2.name": "gemini-2.5-flash"},
        event={"name": "metadata"},
    )
    monocle_trace_asserter.has_scope("agentic.session", session_id)

    # Hosting. The project segment may be the id or the number, so match the rest.
    _, location, engine_id = _RESOURCE_RE.match(AGENT_ENGINE_RESOURCE_NAME).groups()
    engine_suffix = f"/locations/{location}/reasoningEngines/{engine_id}"
    monocle_trace_asserter.where(
        attribute={"span.type": "workflow", "entity.2.type": "app_hosting.gcp_agent_engine"},
        predicate=lambda span: str(span.attributes.get("entity.2.name", "")).startswith("projects/")
        and str(span.attributes.get("entity.2.name", "")).endswith(engine_suffix),
        message=f"no workflow span hosted on Agent Engine '{AGENT_ENGINE_RESOURCE_NAME}'",
    )


if __name__ == "__main__":
    pytest.main([__file__])
