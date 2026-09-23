"""Unit tests for the A2A runner.

A MockTransport serves the agent card and the JSON-RPC answers, so these run
the real a2a SDK client without a server.
"""
import json

import httpx
import pytest

from monocle_test_tools.runner.a2a_runner import A2ARunner
from monocle_test_tools.runner.runner import AgentTypes, get_agent_runner

BASE_URL = "http://agent.test"

AGENT_CARD = {
    "capabilities": {"streaming": False},
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "description": "Converts currencies",
    "name": "currency_agent",
    "preferredTransport": "JSONRPC",
    "protocolVersion": "0.3.0",
    "skills": [],
    "url": f"{BASE_URL}/",
    "version": "1.0",
}

def _task_result(request_id, text="10 USD is 830 INR", task_id="task-1",
                 context_id="ctx-1", state="completed"):
    return {
        "id": request_id,
        "jsonrpc": "2.0",
        "result": {
            "id": task_id,
            "contextId": context_id,
            "kind": "task",
            "status": {"state": state},
            "artifacts": [{"artifactId": "art-1",
                           "parts": [{"kind": "text", "text": text}]}],
        },
    }


def _open_task_result(request_id):
    """A task still waiting on the user, so the next message continues it."""
    return _task_result(request_id, text="Which currency?", state="input-required")


class FakeA2AServer:
    """Serves the agent card and one JSON-RPC answer per message.

    Records every request, so a test can check what the runner put on the wire.
    """

    def __init__(self, *, answer=_task_result):
        self.answer = answer
        self.requests = []
        self.sent_messages = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=AGENT_CARD)

        body = json.loads(request.content)
        self.sent_messages.append(body["params"]["message"])
        payload = json.dumps(self.answer(body["id"])).encode("utf-8")
        return httpx.Response(200, content=payload,
                              headers={"content-type": "application/json"})


# -- registration ----------------------------------------------------------

def test_agent_type_mapping():
    assert isinstance(get_agent_runner(AgentTypes.A2A), A2ARunner)
    assert AgentTypes.A2A == "a2a"


# -- message shaping -------------------------------------------------------

def test_string_message_becomes_a_text_part():
    params = A2ARunner._build_params("how much is 10 USD in INR?", None, None)
    assert params["message"]["role"] == "user"
    assert params["message"]["parts"] == [{"kind": "text", "text": "how much is 10 USD in INR?"}]
    assert params["message"]["messageId"]


def test_params_shaped_dict_is_used_verbatim():
    given = {"message": {"role": "agent", "parts": [{"kind": "text", "text": "hi"}],
                         "messageId": "fixed"},
             "configuration": {"blocking": True}}
    params = A2ARunner._build_params(given, None, None)
    assert params["configuration"] == {"blocking": True}
    assert params["message"]["role"] == "agent"
    assert params["message"]["messageId"] == "fixed"


def test_message_shaped_dict_is_wrapped():
    params = A2ARunner._build_params(
        {"role": "user", "parts": [{"kind": "text", "text": "hi"}]}, None, None)
    assert params["message"]["parts"] == [{"kind": "text", "text": "hi"}]


def test_continuation_ids_are_attached_but_never_override_the_caller():
    params = A2ARunner._build_params("hi", "task-1", "ctx-1")
    assert (params["message"]["taskId"], params["message"]["contextId"]) == ("task-1", "ctx-1")

    given = {"role": "user", "parts": [], "taskId": "caller-task"}
    params = A2ARunner._build_params(given, "task-1", "ctx-1")
    assert params["message"]["taskId"] == "caller-task"


# -- running ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_agent_sends_message_and_returns_response():
    server = FakeA2AServer()
    runner = A2ARunner(transport=server.transport())

    response = await runner.run_agent_async(BASE_URL, "how much is 10 USD in INR?")

    assert response.root.result.artifacts[0].parts[0].root.text == "10 USD is 830 INR"
    assert server.sent_messages[0]["parts"][0]["text"] == "how much is 10 USD in INR?"
    # The card was fetched from the server's well-known path.
    assert server.requests[0].url.path == "/.well-known/agent-card.json"


@pytest.mark.asyncio
async def test_supplied_agent_card_skips_the_card_fetch():
    from a2a.types import AgentCard

    server = FakeA2AServer()
    runner = A2ARunner(agent_card=AgentCard.model_validate(AGENT_CARD),
                       transport=server.transport())

    await runner.run_agent_async(BASE_URL, "hi")

    assert [r.method for r in server.requests] == ["POST"]


@pytest.mark.asyncio
async def test_missing_message_is_rejected():
    runner = A2ARunner(transport=FakeA2AServer().transport())
    with pytest.raises(ValueError):
        await runner.run_agent_async(BASE_URL)


def test_run_agent_sync_wrapper():
    server = FakeA2AServer()
    runner = A2ARunner(transport=server.transport())

    response = runner.run_agent(BASE_URL, "hi")

    assert response.root.result.status.state.value == "completed"


# -- multi-turn ------------------------------------------------------------

@pytest.mark.asyncio
async def test_second_turn_continues_a_task_still_waiting_on_the_user():
    server = FakeA2AServer(answer=_open_task_result)
    runner = A2ARunner(transport=server.transport())

    await runner.run_agent_async(BASE_URL, "How much is 1 USD?", session_id="s1")
    await runner.run_agent_async(BASE_URL, "CAD", session_id="s1")

    first, second = server.sent_messages
    assert "taskId" not in first
    assert (second["taskId"], second["contextId"]) == ("task-1", "ctx-1")


@pytest.mark.asyncio
async def test_a_finished_task_is_not_continued():
    """A finished task rejects new messages, so only the context carries on."""
    server = FakeA2AServer()          # answers with state "completed"
    runner = A2ARunner(transport=server.transport())

    await runner.run_agent_async(BASE_URL, "How much is 1 USD?", session_id="s1")
    await runner.run_agent_async(BASE_URL, "and in CAD?", session_id="s1")

    second = server.sent_messages[1]
    assert "taskId" not in second
    assert second["contextId"] == "ctx-1"


@pytest.mark.asyncio
async def test_sessions_do_not_leak_into_each_other():
    server = FakeA2AServer(answer=_open_task_result)
    runner = A2ARunner(transport=server.transport())

    await runner.run_agent_async(BASE_URL, "first", session_id="s1")
    await runner.run_agent_async(BASE_URL, "other session", session_id="s2")

    second = server.sent_messages[1]
    assert "taskId" not in second and "contextId" not in second


@pytest.mark.asyncio
async def test_end_session_drops_the_continuation_ids():
    server = FakeA2AServer(answer=_open_task_result)
    runner = A2ARunner(transport=server.transport())

    await runner.run_agent_async(BASE_URL, "first", session_id="s1")
    await runner.end_session("s1")
    await runner.run_agent_async(BASE_URL, "second", session_id="s1")

    second = server.sent_messages[1]
    assert "taskId" not in second and "contextId" not in second


@pytest.mark.asyncio
async def test_an_error_answer_leaves_the_session_alone():
    """An error answer has no ids, so the ones already held stay."""
    answers = [_open_task_result,
               lambda request_id: {"id": request_id, "jsonrpc": "2.0",
                                   "error": {"code": -32602, "message": "bad request"}}]

    server = FakeA2AServer(answer=lambda request_id: answers[min(
        len(server.sent_messages) - 1, len(answers) - 1)](request_id))
    runner = A2ARunner(transport=server.transport())

    await runner.run_agent_async(BASE_URL, "first", session_id="s1")
    await runner.run_agent_async(BASE_URL, "second", session_id="s1")
    await runner.run_agent_async(BASE_URL, "third", session_id="s1")

    assert server.sent_messages[2]["taskId"] == "task-1"


@pytest.mark.asyncio
async def test_message_result_is_remembered_too():
    """An agent answering with a Message still names the context to continue."""
    def message_answer(request_id):
        return {"id": request_id, "jsonrpc": "2.0",
                "result": {"kind": "message", "role": "agent", "messageId": "m1",
                           "taskId": "task-9", "contextId": "ctx-9",
                           "parts": [{"kind": "text", "text": "830 INR"}]}}

    server = FakeA2AServer(answer=message_answer)
    runner = A2ARunner(transport=server.transport())

    await runner.run_agent_async(BASE_URL, "first", session_id="s1")
    await runner.run_agent_async(BASE_URL, "second", session_id="s1")

    assert server.sent_messages[1]["contextId"] == "ctx-9"


# -- trace sources --------------------------------------------------------

# -- where the agent's spans come from -------------------------------------

def test_pulls_from_okahu_when_the_workflow_is_known(monkeypatch):
    monkeypatch.delenv("A2A_TRACE_WORKFLOW", raising=False)
    runner = A2ARunner(trace_workflow_name="currency_agent")

    assert runner.get_remote_traces_source() == "okahu"
    # No id: the agent's spans are in the test's trace, which the validator
    # resolves from the local spans.
    assert runner.get_remote_trace_query() == {"workflow_name": "currency_agent"}


def test_takes_the_workflow_from_the_environment(monkeypatch):
    monkeypatch.setenv("A2A_TRACE_WORKFLOW", "from-env")
    assert A2ARunner().get_remote_trace_query() == {"workflow_name": "from-env"}


def test_fetches_nothing_without_a_workflow(monkeypatch):
    """No workflow, nothing to ask Okahu for: the test reads the trace file."""
    monkeypatch.delenv("A2A_TRACE_WORKFLOW", raising=False)
    runner = A2ARunner()
    assert runner.get_remote_traces_source() is None
    assert runner.get_remote_trace_query() == {}

