"""Unit tests for the Foundry runner.

All tests use an injected fake credential and a patched ``requests.post``, so
they run without azure-identity, Azure credentials, a deployed agent, or any
network access.
"""
import pytest
import requests

from monocle_test_tools.runner.runner import get_agent_runner, AgentTypes
from monocle_test_tools.runner.foundry_runner import (
    FOUNDRY_FEATURES_HEADER,
    FOUNDRY_SCOPE,
    FoundryRunner,
)

PROJECT = "https://acct.services.ai.azure.com/api/projects/proj"
AGENT_URL = f"{PROJECT}/agents/travel-agent/endpoint/protocols/openai/responses"
# Workflow the deployed agent reports under — not the test's own workflow.
REMOTE_WORKFLOW = "deployed_agent_workflow"
TRACE_ID = "1004ac724700a70b0c8ab9a1f0fe8234"
SESSION_ID = "0b6e34f11344a024f377e3c346c556286c0665614161cc954ddd5974fd10fa9"


class FakeToken:
    def __init__(self, token="fake-token"):
        self.token = token


class FakeCredential:
    """Records the scope it was asked for and returns a canned token."""

    def __init__(self, token="fake-token"):
        self._token = token
        self.scopes = []

    def get_token(self, *scopes, **kwargs):
        self.scopes.append(scopes)
        return FakeToken(self._token)


class FakeResponse:
    def __init__(self, status_code=200, headers=None, text='{"output": []}'):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)


@pytest.fixture
def captured_post(monkeypatch):
    """Patch requests.post and record the call it receives."""
    calls = []
    response_holder = {
        "response": FakeResponse(
            headers={"x-request-id": TRACE_ID, "x-agent-session-id": SESSION_ID}
        )
    }

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return response_holder["response"]

    monkeypatch.setattr(requests, "post", fake_post)
    return calls, response_holder


# -- registry ---------------------------------------------------------------

def test_agent_type_mapping():
    assert AgentTypes.FOUNDRY.value == "foundry"
    assert isinstance(get_agent_runner("foundry"), FoundryRunner)


# -- url handling -----------------------------------------------------------

def test_api_version_is_added_when_missing():
    assert FoundryRunner._with_api_version(AGENT_URL) == f"{AGENT_URL}?api-version=v1"


def test_api_version_supplied_by_the_caller_is_kept():
    url = f"{AGENT_URL}?api-version=2025-11-15-preview"
    assert FoundryRunner._with_api_version(url) == url


def test_api_version_is_appended_to_an_existing_query():
    assert FoundryRunner._with_api_version(f"{AGENT_URL}?foo=bar") == (
        f"{AGENT_URL}?foo=bar&api-version=v1"
    )


def test_agent_name_is_read_off_the_url():
    assert FoundryRunner._agent_name_from_url(AGENT_URL) == "travel-agent"


def test_agent_name_is_none_when_the_url_names_no_agent():
    assert FoundryRunner._agent_name_from_url(f"{PROJECT}/openai/v1/responses") is None


# -- headers ----------------------------------------------------------------

def test_duplicated_request_id_header_yields_one_id():
    """Foundry returns x-request-id as '<id>,<id>'; the joined form is not an id."""
    assert FoundryRunner._first_header_value(f"{TRACE_ID},{TRACE_ID}") == TRACE_ID


def test_single_valued_request_id_header_is_unchanged():
    assert FoundryRunner._first_header_value(TRACE_ID) == TRACE_ID


@pytest.mark.parametrize("value", [None, "", ",", "  "])
def test_absent_or_empty_request_id_header_yields_none(value):
    assert FoundryRunner._first_header_value(value) is None


# -- payload ----------------------------------------------------------------

def test_string_message_becomes_a_non_streaming_responses_body():
    assert FoundryRunner._build_payload("Book a flight", "travel-agent") == {
        "input": "Book a flight",
        "stream": False,
        "model": "travel-agent",
    }


def test_dict_message_is_sent_verbatim():
    body = {"input": [{"role": "user", "content": []}], "stream": True}
    assert FoundryRunner._build_payload(body, "travel-agent") == body


# -- the request ------------------------------------------------------------

def test_request_carries_token_features_header_and_body(captured_post):
    calls, _ = captured_post
    credential = FakeCredential()
    runner = FoundryRunner(credential=credential, trace_workflow_name=REMOTE_WORKFLOW)

    runner.run_agent(AGENT_URL, "Book a flight from Delhi to Phuket")

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == f"{AGENT_URL}?api-version=v1"
    assert call["headers"]["Authorization"] == "Bearer fake-token"
    assert call["headers"][FOUNDRY_FEATURES_HEADER]
    assert call["json"]["model"] == "travel-agent"
    assert call["json"]["stream"] is False
    assert credential.scopes == [(FOUNDRY_SCOPE,)]


def test_caller_headers_win_over_the_defaults(captured_post):
    calls, _ = captured_post
    runner = FoundryRunner(credential=FakeCredential())

    runner.run_agent(AGENT_URL, "hi", headers={FOUNDRY_FEATURES_HEADER: "Custom=V1"})

    assert calls[0]["headers"][FOUNDRY_FEATURES_HEADER] == "Custom=V1"


def test_response_identifiers_are_captured(captured_post):
    runner = FoundryRunner(credential=FakeCredential(), trace_workflow_name=REMOTE_WORKFLOW)

    runner.run_agent(AGENT_URL, "hi")

    assert runner.get_last_trace_id() == TRACE_ID
    assert runner.get_last_session_id() == SESSION_ID


def test_trace_id_is_captured_even_when_the_call_fails(captured_post):
    """A failing call must still say which trace to go and look at."""
    _, holder = captured_post
    holder["response"] = FakeResponse(status_code=500, headers={"x-request-id": TRACE_ID},
                                      text="boom")
    runner = FoundryRunner(credential=FakeCredential(), trace_workflow_name=REMOTE_WORKFLOW)

    with pytest.raises(requests.HTTPError):
        runner.run_agent(AGENT_URL, "hi")

    assert runner.get_last_trace_id() == TRACE_ID


def test_identifiers_do_not_leak_between_calls(captured_post):
    _, holder = captured_post
    runner = FoundryRunner(credential=FakeCredential(), trace_workflow_name=REMOTE_WORKFLOW)
    runner.run_agent(AGENT_URL, "hi")
    assert runner.get_last_trace_id() == TRACE_ID

    holder["response"] = FakeResponse(headers={})
    runner.run_agent(AGENT_URL, "hi")

    assert runner.get_last_trace_id() is None
    assert runner.get_remote_trace_query() == {}


def test_root_agent_must_be_a_url():
    with pytest.raises(ValueError, match="Responses URL"):
        FoundryRunner(credential=FakeCredential()).run_agent(None, "hi")


# -- remote trace retrieval -------------------------------------------------

def test_remote_trace_query_names_the_trace_id(captured_post):
    runner = FoundryRunner(credential=FakeCredential(), trace_workflow_name=REMOTE_WORKFLOW)

    runner.run_agent(AGENT_URL, "hi")

    assert runner.get_remote_traces_source() == "okahu"
    assert runner.get_remote_trace_query() == {
        "id": TRACE_ID,
        "fact_name": "trace",
        "workflow_name": REMOTE_WORKFLOW,
    }


def test_retrieval_is_skipped_without_a_workflow_name(captured_post):
    """Without somewhere to look, retrieval is skipped rather than importing nothing."""
    runner = FoundryRunner(credential=FakeCredential(), trace_workflow_name=None)

    runner.run_agent(AGENT_URL, "hi")

    assert runner.get_remote_traces_source() is None
    assert runner.get_remote_trace_query() == {}


def test_workflow_name_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("FOUNDRY_TRACE_WORKFLOW", REMOTE_WORKFLOW)
    assert FoundryRunner()._trace_workflow_name == REMOTE_WORKFLOW


def test_retrieval_is_skipped_before_any_call():
    runner = FoundryRunner(credential=FakeCredential(), trace_workflow_name=REMOTE_WORKFLOW)

    assert runner.get_remote_traces_source() is None
    assert runner.get_remote_trace_query() == {}


if __name__ == "__main__":
    pytest.main([__file__])
