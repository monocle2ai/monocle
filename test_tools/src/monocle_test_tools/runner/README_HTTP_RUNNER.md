# HTTP Runner

The HTTP runner drives an agent that is exposed over HTTP: `MonocleValidator`
sends a real HTTP request to your service and then asserts against the traces
that request produced. It is selected by the `agent_type` string passed to
`run_agent` / `test_agent`.

There are two HTTP runners:

| `agent_type` | Class | How server-side spans reach the test |
|---|---|---|
| `"http"` | `HttpRunner` | **Piggybacked on the HTTP response** — the instrumented server appends its spans to the response body, and the runner ingests them locally. No cloud dependency. |
| `"http_with_okahu"` | `HttpOkahuRunner` | Fetched from **Okahu** after the request (the server exports spans to Okahu; the runner pulls them back by trace/scope). |

Both send the request with the [`requests`](https://requests.readthedocs.io)
library and return the `requests.Response`.

---

## `run_agent` API

```python
run_agent(root_agent, agent_type, *args, **kwargs) -> requests.Response
```

- **`root_agent`** — the target **URL string** (e.g. `"http://localhost:8000/chat"`).
  For the HTTP runners this is the request URL, not a callable.
- **`agent_type`** — `"http"` or `"http_with_okahu"`.
- **`**kwargs`** — forwarded verbatim to `requests.request(**kwargs)`, so any
  `requests` argument works: `method`, `json`, `data`, `headers`, `params`,
  `timeout`, `auth`, `cookies`, `files`, `verify`, etc. The runner sets `url`
  for you from `root_agent`.

The call raises `requests.HTTPError` for non-2xx responses
(`response.raise_for_status()` is applied), and returns the `Response`
otherwise. An async variant, `run_agent_async(...)`, is also available.

### Minimal example (pytest, fluent API)

The `monocle_trace_asserter` fixture (registered by the `monocle_test_tools`
pytest plugin) exposes `run_agent` plus the chainable trace assertions:

```python
# conftest.py (only needed when running without installing the package)
pytest_plugins = ["monocle_test_tools.pytest_plugin"]
```

```python
def test_chat_agent(monocle_trace_asserter):
    response = monocle_trace_asserter.run_agent(
        "http://localhost:8000/chat",   # root_agent (URL)
        "http",                          # agent_type
        method="POST",
        json={"q": "What is the capital of France?"},
    )

    # 1) assert on the HTTP response itself
    assert response.status_code == 200
    assert "Paris" in response.json()["answer"]

    # 2) assert on the SERVER-side traces returned in-band
    #    (fluent selectors/assertions run against the piggybacked spans)
    monocle_trace_asserter \
        .called_tool("lookup_capital") \
        .contains_output("Paris")
```

`run_agent` returns the response, and — for `agent_type="http"` — the server's
spans are loaded into the validator so the fluent assertions
(`called_tool`, `called_agent`, `has_output`, `contains_output`, `has_scope`,
`check_eval`, token/duration limits, …) operate on them.

### Direct (non-fixture) usage

```python
from monocle_test_tools.validator import MonocleValidator

validator = MonocleValidator()
response = validator.run_agent(
    "http://localhost:8000/chat", "http",
    method="POST", json={"q": "hi"},
)
```

---

## Configuration

### Client side (the process running the test / `HttpRunner`)

| Env var | Applies to | Purpose |
|---|---|---|
| `MONOCLE_TRACE_RETRIEVAL_KEY` | `"http"` | Retrieval key. When set, `HttpRunner` auto-injects `x-monocle-retrieve-traces: <key>` into the request headers (unless you already supplied that header — caller wins). This is how the test authorizes itself to receive the server's spans. |
| `OKAHU_API_KEY`, `OKAHU_API_ENDPOINT` | `"http_with_okahu"` | Credentials/endpoint used to pull traces back from Okahu. |

The retrieval-key header is injected **only** by `HttpRunner` (a deliberate test
driver aimed at a known server) — never by the generic `requests`
instrumentation, so the key is not broadcast to every host your app calls.

### Server side (the instrumented HTTP service under test)

For `agent_type="http"`, the server must opt in to returning its spans. This is
a two-layer, safe-by-default gate — the feature is completely inert unless
**both** the master switch is on **and** the request is authorized.

| Env var | Purpose |
|---|---|
| `MONOCLE_ENABLE_TRACE_RETURN` | Master switch. Set to `true` to enable returning spans on the response. When unset/`false`, nothing is buffered or appended — no trace leak. |
| `MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY` | Secret key for the built-in authorization callback. A request is authorized when its `x-monocle-retrieve-traces` header value equals this key (constant-time compare). If unset, the default callback denies every request. |
| `MONOCLE_TRACE_RETRIEVAL_CALLBACK` | Optional custom authorization callback, format `"pkg.module:callable"`. It receives the request header dict and returns `bool`. If it can't be loaded (bad spec, import error, not callable) or it raises, the request is **denied**. When set, it replaces the default key callback. |

> Note: `MONOCLE_ENABLE_TRACE_RETURN`, `MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY`, and
> the client-side `MONOCLE_TRACE_RETRIEVAL_KEY` must be set **before**
> `MonocleValidator()` / `setup_monocle_telemetry()` is first constructed in the
> process — the trace-return exporter is wired once at instrumentor-setup time.
> In tests that run an in-process server, set them at module import time.

#### Custom authorization callback

```python
# my_pkg/auth.py
def authorize_trace_retrieval(headers: dict) -> bool:
    # headers is the incoming HTTP header dict; return True to return spans
    return headers.get("x-internal-caller") == "ci" and _valid_token(headers)
```

```bash
export MONOCLE_ENABLE_TRACE_RETURN=true
export MONOCLE_TRACE_RETRIEVAL_CALLBACK="my_pkg.auth:authorize_trace_retrieval"
```

Every failure path (callback unloadable, callback raises, key unset/mismatch,
header absent) denies safely: no scope is set, so the server never captures or
returns spans for that request.

### Related general configuration

The usual Monocle knobs also apply (see the repo `CLAUDE.md`): `MONOCLE_EXPORTER`
selects exporters, `MONOCLE_TEST_WORKFLOW_NAME` tags test traces, etc.

---

## How `agent_type="http"` works (span piggyback)

```
client run_agent(url, "http", ...)
  HttpRunner injects x-monocle-retrieve-traces: <MONOCLE_TRACE_RETRIEVAL_KEY>
  -> HTTP request
       server: authorized? (enabled AND callback(headers)) -> tag spans
       -> route runs, child spans complete
       -> server appends spans as a trailer after the response body,
          announced via the x-monocle-traces response header
  <- HTTP response
  HttpRunner strips the trailer, restores the clean body, and loads the
  server spans into the validator for assertions
```

The `requests.Response` you get back has a clean body (the trailer is removed).

## Framework support

Server-side span return is implemented, behind the same two-layer
authorization gate, for:

| Framework | Buffered response | Streaming response |
|---|---|---|
| FastAPI (ASGI) | Yes | Yes — trailer appended to the ASGI `http.response.body` chain |
| Flask (WSGI) | Yes | Yes — trailer appended via a wrapped WSGI `app_iter` / `start_response` |
| aiohttp | Yes — `web.Response` | Yes — `web.StreamResponse` (`prepare`/`write_eof` hooks) |
| AWS Lambda\* | Yes — trailer appended to the `body` string, header added to the `headers` dict | N/A (Lambda responses are always a single buffered dict; no streaming response model) |
| Azure Functions | Yes — trailer appended to `func.HttpResponse`'s body, header added to `.headers` | N/A (buffered `HttpResponse` model; no streaming response type) |

All five frameworks share the same wire format (`x-monocle-traces` response
header + trailer appended after the body, stripped by the client-side
`RequestSpanHandler`) and the same server-side authorization gate
(`MONOCLE_ENABLE_TRACE_RETURN` + `MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY` /
`MONOCLE_TRACE_RETRIEVAL_CALLBACK`) described above.

\* See the AWS Lambda keyword-invocation caveat under "Known limitations"
below — the table above describes the mechanism, not an unconditional
guarantee for real Lambda deployments.

End-to-end coverage: Flask and aiohttp are driven through a real in-process
server via `HttpRunner`
([`test_http_trace_return_frameworks.py`](../../../tests/integration/test_http_trace_return_frameworks.py)),
mirroring the FastAPI e2e
([`test_http_trace_return.py`](../../../tests/integration/test_http_trace_return.py)).
AWS Lambda and Azure Functions have no real hosting runtime available in
tests, so their e2e coverage drives the route decorator
(`monocle_trace_lambda_function_route` / `monocle_trace_azure_function_route`)
directly with a fake `event`/`req`, and asserts the returned dict /
`HttpResponse` carries the trailer and that stripping it client-side recovers
the span JSON — same file. Streaming for Flask/aiohttp is unit-covered
(`apptrace/tests/unit/test_flask_trace_return.py`,
`test_aiohttp_trace_return.py`) rather than re-proven end-to-end, since it
needs no real server to demonstrate correctly.

## Known limitations

- Only **completed child spans** of the request are returned; the server's root
  HTTP/workflow span ends after the response is sent, so it is not included.
- Client-side handling assumes a **buffered** response (the default for
  `requests`); streaming *consumption on the client* is not covered.
- Designed for direct client→server test setups (no intermediary proxy).
- **AWS Lambda trace-return currently only activates when the handler is
  invoked with keyword arguments** (`event=`/`context=`); under standard
  positional AWS invocation (`handler(event, context)`) the underlying
  `lambda_func_pre_tracing` header extraction (`kwargs['event']`) finds no
  event, so the trace-return scope is never set, no spans are captured, and no
  trailer is returned — the feature silently no-ops rather than erroring.
  This is a pre-existing limitation of the `lambdafunc` instrumentation's
  header-extraction convention (its other accessors, e.g. `get_url`/`get_route`,
  read the event positionally instead, so the two are inconsistent with each
  other), not something introduced by trace-return; it is tracked as a known
  gap rather than fixed here. Azure Functions is unaffected — its Python
  worker invokes handlers with `req=` as a keyword argument, matching what
  `azure_func_pre_tracing` expects.
- **aiohttp streaming (`web.StreamResponse`) trace-return depends on the
  Monocle span context still being active when `write_eof` runs.** If the
  request span has already closed by then, the trailer is skipped and the
  client receives the announced header with no spans — safe (clean body, no
  error), just no trace payload. Buffered aiohttp `web.Response` is unaffected.

## Source

- Runner: [`http_runner.py`](http_runner.py) (`HttpRunner`, `HttpOkahuRunner`)
- Runner registry / `AgentTypes`: [`runner.py`](runner.py)
- Server-side codec, authorization, exporter (in `monocle_apptrace`):
  `instrumentation/common/trace_return.py`, `exporters/trace_return_exporter.py`
- Per-framework injection (in `monocle_apptrace`):
  `instrumentation/metamodel/fastapi/_helper.py`,
  `instrumentation/metamodel/flask/_helper.py`,
  `instrumentation/metamodel/aiohttp/_helper.py`,
  `instrumentation/metamodel/lambdafunc/_helper.py` + `wrapper.py`,
  `instrumentation/metamodel/azfunc/_helper.py` + `wrapper.py`
