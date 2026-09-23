# A2A Runner

The A2A runner tests an agent that speaks the **A2A protocol**. The test acts as
an A2A client — it resolves the agent card, sends a message, gets the response —
and then asserts on the traces that call produced, on both sides of it. Pick it
with the `agent_type` string passed to `run_agent` / `test_agent`.

It sends the call with the [a2a SDK](https://github.com/a2aproject/a2a-python)
(`A2ACardResolver` + `A2AClient`) and returns its `SendMessageResponse`. Monocle
instruments `A2AClient.send_message`, so the test's own call is traced too.

Where the **agent's** spans come from depends on one setting:

| `A2A_TRACE_WORKFLOW` | What the runner does |
|---|---|
| set | Pulls the agent's spans from Okahu after the call. |
| unset | Fetches nothing; the test reads the agent's [trace file](#reading-the-agents-spans-from-a-file). |

The call carries the trace context outward, so **the agent's spans land in the
test's trace**. That is what both lookups are keyed on, and it is why the Okahu
pull needs only a workflow name.

Install the SDK with the extra:

```bash
pip install "monocle_test_tools[a2a]"
```

---

## `run_agent` API

```python
run_agent(root_agent, agent_type, message, **kwargs) -> SendMessageResponse
```

- **`root_agent`** — the agent's **base URL** (e.g. `"http://localhost:10000"`),
  or a ready-made `AgentCard`. Given a URL, the card is fetched from the
  well-known path on each run.
- **`agent_type`** — `"a2a"`.
- **`message`** — usually a string, which becomes a single text part. A dict
  holding `"message"` is sent as `MessageSendParams` unchanged; any other dict
  is taken as the message itself.
- **`session_id`**, **`task_id`**, **`context_id`** — see [multi-turn](#multi-turn).
- **`http_kwargs`** — passed to `A2AClient.send_message`.

`run_agent_async(...)` is the async form, and is what multi-turn tests use.

### Example (pytest, fluent API)

```python
# conftest.py — only needed when the package is not installed
pytest_plugins = ["monocle_test_tools.pytest_plugin"]
```

```python
def test_currency_agent(monocle_trace_asserter):
    response = monocle_trace_asserter.run_agent(
        "http://localhost:10000",          # root_agent (base URL)
        "a2a",                             # agent_type
        "how much is 10 USD in INR?",      # message
    )

    # the A2A response
    assert response.root.result.status.state.value == "completed"

    # the agent's own spans, pulled from Okahu (A2A_TRACE_WORKFLOW is set)
    monocle_trace_asserter.called_tool("get_exchange_rate")
    monocle_trace_asserter.contains_output("INR")
```

### Without the fixture

```python
from monocle_test_tools.validator import MonocleValidator

validator = MonocleValidator()
response = validator.run_agent("http://localhost:10000", "a2a",
                               "how much is 10 USD in INR?")
```

---

### What the A2A call records

The call itself is an `agentic.invocation` span naming the agent it reached:
`entity.1.name` from the agent card, `entity.1.type` `agent2agent.server`,
`entity.1.url`, and `entity.1.method`. Its `data.output` event carries the
answer, the task state, and the `task_id` / `context_id` the call belonged to —
so a trace shows which conversation it was part of.

When the agent is instrumented too, one call leaves **two** named invocation
spans: the hop out (`span.subtype` `routing`) and the agent's own run
(`content_processing`). Both are true, so a count assertion should say which
side it means:

```python
asserter.where(attribute={"span.subtype": "content_processing"}) \
        .called_agent("travel_agent", count=1)
```

## Multi-turn

A2A continues a conversation through the `contextId` the agent answered with,
and an unfinished task through its `taskId`. The runner carries both forward.
Use the multi-turn API, which keeps one runner across the turns:

```python
case = MultiTurnTestCase(
    session_id="rates",
    turns=[{"test_input": ["How much is the exchange rate for 1 USD?"]},
           {"test_input": ["CAD"]}],
)
await validator.test_multi_turn_agent_async(url, "a2a", case)
```

Turn 2 goes out with turn 1's `contextId`, so the agent reads "CAD" as a
follow-up. The `taskId` is carried only while the task still accepts messages
(`submitted`, `working`, `input-required`, `auth-required`) — so an agent that
asked a question is answered in place. A `completed`, `canceled`, `failed` or
`rejected` task rejects further messages, so the runner drops it and the
conversation carries on through the context alone.

A loop of `run_agent_async(..., session_id=...)` threads the ids too — turns
sharing a session id share the runner that holds them. Ids you set in the
message body are never overwritten.

---

## Configuration

To pull the agent's spans from Okahu:

| Env var | Purpose |
|---|---|
| `A2A_TRACE_WORKFLOW` | Workflow the **agent** exports its spans under — not the test's. Also settable as `A2ARunner(trace_workflow_name=...)`. Unset means no pull. |
| `OKAHU_API_KEY`, `OKAHU_API_ENDPOINT` | Okahu credentials and endpoint. |

Okahu serves traces only for a workflow that exists in your account, so use the
name the agent is registered under; an unknown name answers 404.

### Reading the agent's spans from a file

An agent that writes trace files needs no runner support. Leave
`A2A_TRACE_WORKFLOW` unset and import the file, pointing `trace_path` at the directory
the agent writes to (its `MONOCLE_TRACE_OUTPUT_PATH`). The test needs read
access to that directory, so this suits a local agent:

```python
response = asserter.run_agent(url, "a2a", "how much is 10 USD in INR?")
asserter.import_traces("file", trace_path="/path/to/agent/.monocle/test_traces",
                       wait_seconds=20)
```

The trace id comes from the call, and the file is found by it. Without
`trace_path` the search runs in the test's own trace directory, which holds
only the test's spans.

The file is written after the response goes out, so pass `wait_seconds=` (or set
`MONOCLE_FILE_TRACE_WAIT`) to let the import wait for it rather than failing on
a trace that is about to appear.

---

## An agent that calls an agent

When the agent under test calls *another* agent over A2A, drive the local agent
as usual (`agent_type="langgraph"`, …) and give its `A2AClient` a traced httpx
client, so the far side joins the same trace:

```python
from monocle_test_tools.a2a_transport import make_traced_httpx_client
from a2a.client import A2AClient

async with make_traced_httpx_client() as httpx_client:
    client = A2AClient(httpx_client=httpx_client, agent_card=card)
    response = await client.send_message(request)
```

The remote agent's spans are then fetched the usual way — from Okahu, or from
its trace file.
