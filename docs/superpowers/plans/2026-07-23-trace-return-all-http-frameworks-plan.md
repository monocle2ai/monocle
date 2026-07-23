# Trace-Return for All HTTP Frameworks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend server-side trace-return response injection from FastAPI to Flask, aiohttp, AWS Lambda, and Azure Functions (buffered for all; streaming for Flask + aiohttp).

**Architecture:** Auth + span capture already work for every framework (all call `extract_http_headers`). Add framework-agnostic trailer helpers in `trace_return.py`, then per-framework response injection: object-mutation for buffered responses, iterable/write-path wrapping for Flask/aiohttp streaming.

**Tech Stack:** Python; OpenTelemetry; Flask/werkzeug (WSGI), aiohttp (async), AWS Lambda (dict), Azure Functions (`func.HttpResponse`); pytest.

## Global Constraints

- Wire format unchanged: response header `x-monocle-traces` = `v1; delim=<uuid>`; trailer = `<delimiter><gzip+base64(spans json)>`.
- Injection happens only when `is_trace_return_enabled()` AND the request was authorized (i.e. spans were captured for this trace — `get_response_trailer`/`pop_and_build_trailer` return `None` otherwise). A denied/disabled request's response is byte-identical (no header, no trailer).
- Trace id for buffered paths comes from the `span` passed to `post_task_processing` (`span.get_span_context().trace_id`). For streaming paths (outside `post_task_processing`), use `get_current_monocle_span().get_span_context().trace_id` — NOT raw `get_current_span()` (Monocle isolates its spans under `MONOCLE_ISOLATE_SPANS=true`).
- apptrace must NOT import from test_tools.
- Do not change the authorization model, exporter, codec, or client side.
- FastAPI behavior must not change.
- TDD: failing test first. Run tests from the repo root (shared `pytest.ini`).

---

### Task 1: Shared trailer helpers in `trace_return.py`

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/common/trace_return.py`
- Modify: `apptrace/src/monocle_apptrace/instrumentation/common/span_handler.py` (`HttpSpanHandler.build_trace_return_trailer` delegates)
- Test: `apptrace/tests/unit/test_trace_return_response_helper.py`

**Interfaces:**
- Produces:
  - `pop_and_build_trailer(trace_id: int, delimiter: str) -> bytes | None`
  - `get_response_trailer(trace_id: int) -> tuple[str, bytes] | None` (returns `(header_value, trailer_bytes)`)
- `HttpSpanHandler.build_trace_return_trailer` now delegates to `pop_and_build_trailer` (FastAPI unchanged).

- [ ] **Step 1: Write the failing test**

Create `apptrace/tests/unit/test_trace_return_response_helper.py`:

```python
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.exporters.trace_return_exporter import get_trace_return_exporter


class FakeCtx:
    def __init__(self, trace_id): self.trace_id = trace_id


class FakeSpan:
    def __init__(self, trace_id):
        self._attributes = {"monocle_apptrace.version": "1.0", "scope.monocle_trace_return": "true"}
        self._ctx = FakeCtx(trace_id)
    @property
    def attributes(self): return self._attributes
    def get_span_context(self): return self._ctx
    def to_json(self): return '{"name": "inference"}'


def test_pop_and_build_trailer_roundtrip():
    exp = get_trace_return_exporter(); exp.clear()
    exp.export([FakeSpan(7)])
    delim = tr.make_delimiter()
    trailer = tr.pop_and_build_trailer(7, delim)
    assert trailer is not None
    clean, payload = tr.split_body_and_trailer(b"BODY" + trailer, delim)
    assert clean == b"BODY"
    import json
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "inference"
    # evicted
    assert tr.pop_and_build_trailer(7, tr.make_delimiter()) is None


def test_get_response_trailer_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    get_trace_return_exporter().clear()
    assert tr.get_response_trailer(7) is None


def test_get_response_trailer_enabled(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear()
    exp.export([FakeSpan(9)])
    result = tr.get_response_trailer(9)
    assert result is not None
    header_value, trailer = result
    assert header_value.startswith("v1;")
    delim = tr.parse_delimiter_from_header(header_value)
    clean, payload = tr.split_body_and_trailer(b"X" + trailer, delim)
    assert clean == b"X"


def test_get_response_trailer_enabled_no_spans(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    get_trace_return_exporter().clear()
    assert tr.get_response_trailer(123456) is None
```

Run: `pytest apptrace/tests/unit/test_trace_return_response_helper.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'pop_and_build_trailer'`

- [ ] **Step 2: Add the module functions**

Append to `trace_return.py`:

```python
def pop_and_build_trailer(trace_id: int, delimiter: str) -> "bytes | None":
    """Pop this trace's captured spans from the exporter and build trailer bytes.
    Returns None when there are no spans to return."""
    from monocle_apptrace.exporters.trace_return_exporter import get_trace_return_exporter
    spans = get_trace_return_exporter().pop_spans_for_trace(trace_id)
    if not spans:
        return None
    return build_trailer_bytes(spans, delimiter)


def get_response_trailer(trace_id: int) -> "tuple[str, bytes] | None":
    """Convenience for buffered injection: make a delimiter, pop+build the
    trailer, and return (response header value, trailer bytes). None when the
    feature is disabled or there are no spans for this trace."""
    if not is_trace_return_enabled():
        return None
    delimiter = make_delimiter()
    trailer = pop_and_build_trailer(trace_id, delimiter)
    if trailer is None:
        return None
    return build_response_header_value(delimiter), trailer
```

- [ ] **Step 3: Make `HttpSpanHandler.build_trace_return_trailer` delegate**

In `span_handler.py`, replace the body of `build_trace_return_trailer` with a delegation:

```python
    def build_trace_return_trailer(self, trace_id: int, delimiter: str) -> "bytes | None":
        """Pop this trace's captured spans and build the response trailer bytes.
        Returns None when there is nothing to return."""
        from monocle_apptrace.instrumentation.common import trace_return as tr
        return tr.pop_and_build_trailer(trace_id, delimiter)
```

- [ ] **Step 4: Run tests + FastAPI regression**

Run:
```
pytest apptrace/tests/unit/test_trace_return_response_helper.py -v
pytest apptrace/tests/unit/test_http_span_handler_trailer.py apptrace/tests/unit/test_fastapi_trace_return_send.py -v
```
Expected: new helper tests PASS (4); existing trailer + FastAPI send tests still PASS (unchanged behavior).

- [ ] **Step 5: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/common/trace_return.py apptrace/src/monocle_apptrace/instrumentation/common/span_handler.py apptrace/tests/unit/test_trace_return_response_helper.py
git commit -m "feat(apptrace): shared trace-return trailer helpers (pop_and_build_trailer, get_response_trailer)"
```

---

### Task 2: AWS Lambda buffered injection

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/metamodel/lambdafunc/_helper.py` (`lambdaSpanHandler`)
- Test: `apptrace/tests/unit/test_lambda_trace_return.py`

**Interfaces:**
- Consumes: `get_response_trailer` (Task 1).
- Produces: an authorized Lambda response dict gains `headers["x-monocle-traces"]` and the trailer appended to `body`.

- [ ] **Step 1: Write the failing test**

Create `apptrace/tests/unit/test_lambda_trace_return.py`:

```python
import json
from monocle_apptrace.instrumentation.metamodel.lambdafunc._helper import lambdaSpanHandler
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.exporters.trace_return_exporter import get_trace_return_exporter


class FakeCtx:
    def __init__(self, tid): self.trace_id = tid
class FakeSpan:
    def __init__(self, tid):
        self._a = {"monocle_apptrace.version": "1.0", "scope.monocle_trace_return": "true"}
        self._c = FakeCtx(tid)
    @property
    def attributes(self): return self._a
    def get_span_context(self): return self._c
    def to_json(self): return '{"name": "lambda_child"}'


def test_lambda_injects_trailer(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(11)])
    result = {"statusCode": 200, "headers": {"Content-Type": "application/json"},
              "body": json.dumps({"answer": "hi"})}
    lambdaSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=result, ex=None, span=FakeSpan(11), parent_span=None)
    # header set
    assert any(k.lower() == "x-monocle-traces" for k in result["headers"])
    hv = next(v for k, v in result["headers"].items() if k.lower() == "x-monocle-traces")
    delim = tr.parse_delimiter_from_header(hv)
    clean, payload = tr.split_body_and_trailer(result["body"].encode("utf-8"), delim)
    assert json.loads(clean.decode())["answer"] == "hi"
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "lambda_child"


def test_lambda_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    get_trace_return_exporter().clear()
    result = {"statusCode": 200, "headers": {}, "body": "{}"}
    lambdaSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=result, ex=None, span=FakeSpan(11), parent_span=None)
    assert result == {"statusCode": 200, "headers": {}, "body": "{}"}
```

Run: `pytest apptrace/tests/unit/test_lambda_trace_return.py -v`
Expected: FAIL — the header is not added (base `post_task_processing` does nothing).

- [ ] **Step 2: Add `post_task_processing` to `lambdaSpanHandler`**

In `lambdafunc/_helper.py`, add imports:

```python
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER
```

Add to `class lambdaSpanHandler(SpanHandler):`:

```python
    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        try:
            if isinstance(result, dict) and isinstance(result.get("body"), str):
                trace_id = span.get_span_context().trace_id if span is not None else 0
                payload = tr.get_response_trailer(trace_id)
                if payload is not None:
                    header_value, trailer = payload
                    result["body"] = result["body"] + trailer.decode("ascii")
                    headers = result.get("headers")
                    if not isinstance(headers, dict):
                        headers = {}
                    headers[TRACE_RETURN_RESPONSE_HEADER] = header_value
                    result["headers"] = headers
        except Exception as e:
            logger.debug(f"lambda trace-return injection skipped: {e}")
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)
```

(The trailer is ASCII — `<delimiter><base64>` — so `.decode("ascii")` is safe and keeps `body` a str.)

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest apptrace/tests/unit/test_lambda_trace_return.py -v`
Expected: PASS (2)

- [ ] **Step 4: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/metamodel/lambdafunc/_helper.py apptrace/tests/unit/test_lambda_trace_return.py
git commit -m "feat(apptrace): Lambda trace-return response injection"
```

---

### Task 3: Azure Functions buffered injection

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/metamodel/azfunc/_helper.py` (`azureSpanHandler`)
- Test: `apptrace/tests/unit/test_azfunc_trace_return.py`

**Interfaces:**
- Consumes: `get_response_trailer` (Task 1).
- Produces: an authorized Azure Functions `HttpResponse` carries the header and the trailer appended to its body.

**Constraint:** `post_task_processing` cannot replace the returned object (its return value is discarded by the wrapper). `func.HttpResponse` has no public body setter. So mutate the body **in place** via its internal buffer, which `get_body()` reads and the Azure worker serializes after the function returns.

- [ ] **Step 1: Investigate the HttpResponse body attribute**

Run:
```bash
python -c "import azure.functions as f; r=f.HttpResponse(body=b'abc', status_code=200); print([a for a in vars(r)]); print(r.get_body())"
```
Expected: prints the instance attributes (the private body is name-mangled, typically `_HttpResponse__http_response` or `_HttpResponse__body`) and `b'abc'`. Note the exact attribute holding the raw body bytes; the code below uses `_HttpResponse__body` — adjust to the actual name if different, and confirm `get_body()` reflects a change to it.

- [ ] **Step 2: Write the failing test**

Create `apptrace/tests/unit/test_azfunc_trace_return.py`:

```python
import json
import pytest
azure_functions = pytest.importorskip("azure.functions")
from monocle_apptrace.instrumentation.metamodel.azfunc._helper import azureSpanHandler
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.exporters.trace_return_exporter import get_trace_return_exporter


class FakeCtx:
    def __init__(self, tid): self.trace_id = tid
class FakeSpan:
    def __init__(self, tid):
        self._a = {"monocle_apptrace.version": "1.0", "scope.monocle_trace_return": "true"}
        self._c = FakeCtx(tid)
    @property
    def attributes(self): return self._a
    def get_span_context(self): return self._c
    def to_json(self): return '{"name": "azfunc_child"}'


def test_azfunc_injects_trailer(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(21)])
    resp = azure_functions.HttpResponse(body=json.dumps({"answer": "hi"}).encode("utf-8"),
                                        status_code=200, mimetype="application/json")
    azureSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=resp, ex=None, span=FakeSpan(21), parent_span=None)
    hv = resp.headers.get("x-monocle-traces")
    assert hv is not None
    delim = tr.parse_delimiter_from_header(hv)
    body = resp.get_body()
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert json.loads(clean.decode())["answer"] == "hi"
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "azfunc_child"


def test_azfunc_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    get_trace_return_exporter().clear()
    resp = azure_functions.HttpResponse(body=b"{}", status_code=200)
    azureSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=resp, ex=None, span=FakeSpan(21), parent_span=None)
    assert resp.get_body() == b"{}"
    assert resp.headers.get("x-monocle-traces") is None
```

Run: `pytest apptrace/tests/unit/test_azfunc_trace_return.py -v`
Expected: FAIL (header/body unchanged), or SKIP if azure-functions isn't installed.

- [ ] **Step 3: Add `post_task_processing` to `azureSpanHandler`**

In `azfunc/_helper.py`, add imports:

```python
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER
```

Add to `class azureSpanHandler(SpanHandler):`:

```python
    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        try:
            if hasattr(result, "get_body") and hasattr(result, "headers"):
                trace_id = span.get_span_context().trace_id if span is not None else 0
                payload = tr.get_response_trailer(trace_id)
                if payload is not None:
                    header_value, trailer = payload
                    body = result.get_body() or b""
                    if isinstance(body, str):
                        body = body.encode("utf-8")
                    new_body = body + trailer
                    # func.HttpResponse has no public body setter; mutate the
                    # name-mangled private buffer that get_body() reads. Adjust
                    # the attribute name here to whatever Step 1 found.
                    setattr(result, "_HttpResponse__body", new_body)
                    result.headers[TRACE_RETURN_RESPONSE_HEADER] = header_value
        except Exception as e:
            logger.debug(f"azfunc trace-return injection skipped: {e}")
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)
```

Fallback if the private-buffer mutation does not survive to serialization (Step 1/test shows `get_body()` unchanged): instead add a `response_processor` to `AZFUNC_HTTP_PROCESSOR` that rebuilds a new `HttpResponse(body=old+trailer, status_code=..., headers={**old, header}, mimetype=...)` and returns it (the `response_processor` return value replaces the result; it must also call the provided `finalize(return_value)` for span processing). Document whichever path is used in the report.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest apptrace/tests/unit/test_azfunc_trace_return.py -v`
Expected: PASS (2), or SKIP if azure-functions not installed (note it in the report and, if possible, `pip install azure-functions` in the dev env to actually exercise it).

- [ ] **Step 5: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/metamodel/azfunc/_helper.py apptrace/tests/unit/test_azfunc_trace_return.py
git commit -m "feat(apptrace): Azure Functions trace-return response injection"
```

---

### Task 4: Flask injection (buffered + streaming)

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/metamodel/flask/_helper.py`
- Modify: `apptrace/src/monocle_apptrace/instrumentation/metamodel/flask/methods.py` (custom response wrapper)
- Test: `apptrace/tests/unit/test_flask_trace_return.py`

**Interfaces:**
- Consumes: `get_response_trailer`, `make_delimiter`, `build_response_header_value`, `pop_and_build_trailer`, `is_trace_return_enabled` (Task 1 / trace_return); `is_scope_set`; `get_current_monocle_span`; `TRACE_RETURN_SCOPE_NAME`, `TRACE_RETURN_RESPONSE_HEADER`.
- Produces: authorized Flask responses carry the header + trailer; werkzeug `Response.__call__` is wrapped so both buffered (mutate `Response` before serialization) and streamed (`is_streamed` → wrap the WSGI `app_iter` to append a trailer chunk) are handled.

**Approach:** Replace the `task_wrapper` on `werkzeug.wrappers.response.Response.__call__` with a custom `flask_response_wrapper` (`@with_tracer_wrapper`) that:
1. Determines injection eligibility: `is_trace_return_enabled() and is_scope_set(TRACE_RETURN_SCOPE_NAME)`.
2. If eligible and `not instance.is_streamed`: append the trailer to the response body before span processing (`get_response_trailer(trace_id)`; `instance.set_data(instance.get_data() + trailer)`; werkzeug recomputes Content-Length) and set `instance.headers[HEADER]`. Then call `monocle_wrapper` as normal.
3. If eligible and `instance.is_streamed`: generate a delimiter, set `instance.headers[HEADER] = build_response_header_value(delimiter)`, call `monocle_wrapper` to get the `app_iter`, and return a wrapped iterator that yields all real chunks then one trailer chunk (`pop_and_build_trailer(trace_id, delimiter)`).
4. Otherwise: call `monocle_wrapper` unchanged.

`trace_id` via `get_current_monocle_span().get_span_context().trace_id`.

- [ ] **Step 1: Write the failing test**

Create `apptrace/tests/unit/test_flask_trace_return.py`. It exercises the wrapper's helper logic directly against a real werkzeug Response and a fake streamed response, with the current Monocle span stubbed:

```python
import json
import pytest
werkzeug = pytest.importorskip("werkzeug")
from werkzeug.wrappers import Response
from monocle_apptrace.instrumentation.metamodel.flask import _helper as fh
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.exporters.trace_return_exporter import get_trace_return_exporter


class FakeCtx:
    def __init__(self, tid): self.trace_id = tid
class FakeSpan:
    def __init__(self, tid):
        self._a = {"monocle_apptrace.version": "1.0", "scope.monocle_trace_return": "true"}
        self._c = FakeCtx(tid)
    @property
    def attributes(self): return self._a
    def get_span_context(self): return self._c
    def to_json(self): return '{"name": "flask_child"}'


def test_flask_buffered_injection(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(31)])
    resp = Response(json.dumps({"answer": "hi"}), mimetype="application/json")
    # inject buffered (helper under test)
    fh._flask_inject_buffered(resp, trace_id=31)
    hv = resp.headers.get("x-monocle-traces")
    assert hv is not None
    delim = tr.parse_delimiter_from_header(hv)
    clean, payload = tr.split_body_and_trailer(resp.get_data(), delim)
    assert json.loads(clean.decode())["answer"] == "hi"
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "flask_child"
    # werkzeug recomputed content-length
    assert int(resp.headers["Content-Length"]) == len(resp.get_data())


def test_flask_streaming_injection(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(32)])
    delim = tr.make_delimiter()
    real_iter = iter([b"chunk-a", b"chunk-b"])
    wrapped = list(fh._flask_wrap_stream(real_iter, trace_id=32, delimiter=delim))
    body = b"".join(wrapped)
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert clean == b"chunk-achunk-b"
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "flask_child"


def test_flask_buffered_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    get_trace_return_exporter().clear()
    resp = Response("{}", mimetype="application/json")
    fh._flask_inject_buffered(resp, trace_id=31)
    assert resp.get_data() == b"{}"
    assert resp.headers.get("x-monocle-traces") is None
```

Run: `pytest apptrace/tests/unit/test_flask_trace_return.py -v`
Expected: FAIL — `_flask_inject_buffered` / `_flask_wrap_stream` don't exist.

- [ ] **Step 2: Implement the helpers + custom wrapper**

In `flask/_helper.py`, add imports:

```python
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER, TRACE_RETURN_SCOPE_NAME
from monocle_apptrace.instrumentation.common.utils import is_scope_set, get_current_monocle_span
```

Add the helpers and wrapper:

```python
def _flask_inject_buffered(response, trace_id) -> None:
    payload = tr.get_response_trailer(trace_id)
    if payload is None:
        return
    header_value, trailer = payload
    response.set_data(response.get_data() + trailer)
    response.headers[TRACE_RETURN_RESPONSE_HEADER] = header_value


def _flask_wrap_stream(app_iter, trace_id, delimiter):
    for chunk in app_iter:
        yield chunk
    trailer = tr.pop_and_build_trailer(trace_id, delimiter)
    if trailer:
        yield trailer


@with_tracer_wrapper
def flask_response_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs):
    """Wrap werkzeug Response.__call__ to append the trace-return trailer.
    Buffered: mutate the Response body before it serializes. Streamed: set the
    header and wrap the WSGI app_iter to append a final trailer chunk."""
    eligible = tr.is_trace_return_enabled() and is_scope_set(TRACE_RETURN_SCOPE_NAME)
    if not eligible:
        return monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)
    trace_id = get_current_monocle_span().get_span_context().trace_id
    if not getattr(instance, "is_streamed", False):
        _flask_inject_buffered(instance, trace_id)
        return monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)
    # streamed
    delimiter = tr.make_delimiter()
    instance.headers[TRACE_RETURN_RESPONSE_HEADER] = tr.build_response_header_value(delimiter)
    app_iter = monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)
    return _flask_wrap_stream(app_iter, trace_id, delimiter)
```

In `flask/methods.py`, change the `werkzeug.wrappers.response.Response.__call__` entry's `wrapper_method` from `task_wrapper` to `flask_response_wrapper` (import it), keeping the same `span_handler`/`output_processor`:

```python
from monocle_apptrace.instrumentation.metamodel.flask._helper import flask_task_wrapper, flask_response_wrapper
...
    {
        "package": "werkzeug.wrappers.response",
        "object": "Response",
        "method": "__call__",
        "wrapper_method": flask_response_wrapper,
        "span_handler": "flask_response_handler",
        "output_processor": FLASK_RESPONSE_PROCESSOR,
    }
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest apptrace/tests/unit/test_flask_trace_return.py -v`
Expected: PASS (3)

- [ ] **Step 4: Regression — existing Flask tests**

Run: `pytest apptrace/tests -k flask -v`
Expected: no new failures vs. before (report counts; pre-existing unrelated collection errors are OK).

- [ ] **Step 5: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/metamodel/flask/_helper.py apptrace/src/monocle_apptrace/instrumentation/metamodel/flask/methods.py apptrace/tests/unit/test_flask_trace_return.py
git commit -m "feat(apptrace): Flask trace-return response injection (buffered + streaming)"
```

---

### Task 5: aiohttp injection (buffered + streaming)

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/metamodel/aiohttp/_helper.py`
- Modify: `apptrace/src/monocle_apptrace/instrumentation/metamodel/aiohttp/methods.py` (streaming hooks)
- Test: `apptrace/tests/unit/test_aiohttp_trace_return.py`

**Interfaces:**
- Consumes: Task 1 helpers; `TRACE_RETURN_RESPONSE_HEADER`, `TRACE_RETURN_SCOPE_NAME`; `is_scope_set`; `get_current_monocle_span`.
- Produces: buffered `web.Response` gains header + appended body via `aiohttpSpanHandler.post_task_processing`; streaming `web.StreamResponse` gains the header at `prepare` and a trailer written at `write_eof`.

**Buffered** is straightforward. **Streaming** requires verifying `StreamResponse.prepare`/`write_eof` are wrappable; if not cleanly, degrade aiohttp to buffered-only with a documented limitation.

- [ ] **Step 1: Write the failing test (buffered + streaming helpers)**

Create `apptrace/tests/unit/test_aiohttp_trace_return.py`:

```python
import json
import pytest
aiohttp_web = pytest.importorskip("aiohttp.web")
from monocle_apptrace.instrumentation.metamodel.aiohttp import _helper as ah
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.exporters.trace_return_exporter import get_trace_return_exporter


class FakeCtx:
    def __init__(self, tid): self.trace_id = tid
class FakeSpan:
    def __init__(self, tid):
        self._a = {"monocle_apptrace.version": "1.0", "scope.monocle_trace_return": "true"}
        self._c = FakeCtx(tid)
    @property
    def attributes(self): return self._a
    def get_span_context(self): return self._c
    def to_json(self): return '{"name": "aiohttp_child"}'


def test_aiohttp_buffered_injection(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    exp = get_trace_return_exporter(); exp.clear(); exp.export([FakeSpan(41)])
    resp = aiohttp_web.Response(body=json.dumps({"answer": "hi"}).encode("utf-8"),
                                content_type="application/json")
    ah._aiohttp_inject_buffered(resp, trace_id=41)
    hv = resp.headers.get("x-monocle-traces")
    assert hv is not None
    delim = tr.parse_delimiter_from_header(hv)
    clean, payload = tr.split_body_and_trailer(bytes(resp.body), delim)
    assert json.loads(clean.decode())["answer"] == "hi"
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "aiohttp_child"


def test_aiohttp_buffered_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    get_trace_return_exporter().clear()
    resp = aiohttp_web.Response(body=b"{}", content_type="application/json")
    ah._aiohttp_inject_buffered(resp, trace_id=41)
    assert bytes(resp.body) == b"{}"
    assert resp.headers.get("x-monocle-traces") is None
```

Run: `pytest apptrace/tests/unit/test_aiohttp_trace_return.py -v`
Expected: FAIL — `_aiohttp_inject_buffered` doesn't exist.

- [ ] **Step 2: Implement buffered injection**

In `aiohttp/_helper.py`, add imports:

```python
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER, TRACE_RETURN_SCOPE_NAME
from monocle_apptrace.instrumentation.common.utils import is_scope_set, get_current_monocle_span
```

Add the helper and wire it into `aiohttpSpanHandler.post_task_processing`:

```python
def _aiohttp_inject_buffered(response, trace_id) -> None:
    # Only web.Response (buffered) has a settable .body; StreamResponse does not.
    if not hasattr(response, "body"):
        return
    try:
        current = response.body
    except Exception:
        return
    if current is None:
        return
    payload = tr.get_response_trailer(trace_id)
    if payload is None:
        return
    header_value, trailer = payload
    response.body = bytes(current) + trailer
    response.headers[TRACE_RETURN_RESPONSE_HEADER] = header_value
```

Add to `class aiohttpSpanHandler(HttpSpanHandler):`:

```python
    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        try:
            # Buffered web.Response only; StreamResponse is handled via prepare/write_eof hooks.
            from aiohttp import web
            if isinstance(result, web.Response):
                trace_id = span.get_span_context().trace_id if span is not None else 0
                _aiohttp_inject_buffered(result, trace_id)
        except Exception as e:
            logger.debug(f"aiohttp trace-return buffered injection skipped: {e}")
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)
```

Run: `pytest apptrace/tests/unit/test_aiohttp_trace_return.py -v` → PASS (2).

- [ ] **Step 3: Investigate + implement streaming hooks**

Verify the StreamResponse hooks are wrappable:
```bash
python -c "from aiohttp import web; print(hasattr(web.StreamResponse,'prepare'), hasattr(web.StreamResponse,'write_eof'))"
```
Expected: `True True`.

Add streaming support: a `prepare` wrapper that sets the header before headers are sent, and a `write_eof` wrapper that writes the trailer before EOF. In `aiohttp/_helper.py`:

```python
async def aiohttp_streamresponse_prepare(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs):
    try:
        if tr.is_trace_return_enabled() and is_scope_set(TRACE_RETURN_SCOPE_NAME) and not instance.prepared:
            delimiter = tr.make_delimiter()
            instance._monocle_tr_delimiter = delimiter
            instance.headers[TRACE_RETURN_RESPONSE_HEADER] = tr.build_response_header_value(delimiter)
    except Exception as e:
        logger.debug(f"aiohttp stream prepare header skipped: {e}")
    return await wrapped(*args, **kwargs)


async def aiohttp_streamresponse_write_eof(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs):
    try:
        delimiter = getattr(instance, "_monocle_tr_delimiter", None)
        if delimiter is not None:
            trace_id = get_current_monocle_span().get_span_context().trace_id
            trailer = tr.pop_and_build_trailer(trace_id, delimiter)
            if trailer:
                await instance.write(trailer)
            instance._monocle_tr_delimiter = None
    except Exception as e:
        logger.debug(f"aiohttp stream write_eof trailer skipped: {e}")
    return await wrapped(*args, **kwargs)
```

Both are plain wrappers (not `@with_tracer_wrapper`); register them directly. In `aiohttp/methods.py`, add two entries (no span/output processor — pure behavior wrappers). Because these are not span-creating wrappers, register them via a small adapter matching the instrumentor's `wrapper_method(tracer, handler, method_config)` signature — mirror how `flask_task_wrapper`/`streaming_response_wrapper` are shaped (both use `@with_tracer_wrapper`). If a non-span wrapper shape isn't supported by the method registry, wrap them with `@with_tracer_wrapper` and do NOT call `monocle_wrapper` (like FastAPI's `streaming_response_wrapper`, which calls `wrapped` directly to avoid creating a span). Add:

```python
from monocle_apptrace.instrumentation.metamodel.aiohttp._helper import (
    aiohttp_streamresponse_prepare, aiohttp_streamresponse_write_eof,
)

AIOHTTP_METHODS += [
    {
        "package": "aiohttp.web_response",
        "object": "StreamResponse",
        "method": "prepare",
        "wrapper_method": aiohttp_streamresponse_prepare,
    },
    {
        "package": "aiohttp.web_response",
        "object": "StreamResponse",
        "method": "write_eof",
        "wrapper_method": aiohttp_streamresponse_write_eof,
    },
]
```

Decorate the two functions with `@with_tracer_wrapper` (matching the registry's expected signature). Add a unit test wrapping a fake StreamResponse that records `write()` calls, asserting the trailer is written before EOF and the header set at prepare. If the registry cannot register a behavior-only wrapper cleanly (verify against how `fastapi/_helper.streaming_response_wrapper` is registered), fall back to buffered-only for aiohttp and add a note to the report + the docs limitations (Task 6).

- [ ] **Step 4: Run tests + regression**

Run:
```
pytest apptrace/tests/unit/test_aiohttp_trace_return.py -v
pytest apptrace/tests -k aiohttp -v
```
Expected: aiohttp trace-return tests PASS; existing aiohttp tests show no new failures.

- [ ] **Step 5: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/metamodel/aiohttp/_helper.py apptrace/src/monocle_apptrace/instrumentation/metamodel/aiohttp/methods.py apptrace/tests/unit/test_aiohttp_trace_return.py
git commit -m "feat(apptrace): aiohttp trace-return response injection (buffered + streaming)"
```

---

### Task 6: Integration tests + docs

**Files:**
- Test: `test_tools/tests/integration/test_http_trace_return_frameworks.py`
- Modify: `test_tools/src/monocle_test_tools/runner/README_HTTP_RUNNER.md` (framework support + limitations)

**Interfaces:** consumes everything above; proves end-to-end via `HttpRunner`.

- [ ] **Step 1: Flask e2e (discriminating)**

Create `test_tools/tests/integration/test_http_trace_return_frameworks.py` with a Flask app served by werkzeug in a thread, driven by `HttpRunner`. Set the env at module import (server + client key), mirroring the FastAPI e2e:

```python
import os
os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "true"
os.environ["MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY"] = "fw-s3cret"
os.environ["MONOCLE_TRACE_RETRIEVAL_KEY"] = "fw-s3cret"

import json, socket, threading, time
import pytest
pytest_plugins = ["monocle_test_tools.pytest_plugin"]


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def flask_server():
    flask = pytest.importorskip("flask")
    from werkzeug.serving import make_server
    from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method

    app = flask.Flask(__name__)

    @monocle_trace_method(span_name="answer_question")
    def _answer(q): return f"echo: {q}"

    @app.post("/chat")
    def chat():
        data = flask.request.get_json(force=True, silent=True) or {}
        return {"answer": _answer(data.get("q", ""))}

    port = _free_port()
    srv = make_server("127.0.0.1", port, app)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"
    srv.shutdown(); t.join(timeout=5)


def test_flask_server_spans_returned(flask_server, monocle_trace_asserter):
    response = monocle_trace_asserter.run_agent(
        flask_server + "/chat", "http", method="POST", json={"q": "hi"})
    assert response.json()["answer"] == "echo: hi"
    raw = getattr(response, "_monocle_remote_spans", None)
    assert raw is not None, "no piggybacked spans from Flask server"
    assert "answer_question" in [s.get("name") for s in json.loads(raw)]
```

Run: `pytest test_tools/tests/integration/test_http_trace_return_frameworks.py -v`
Expected: PASS.

- [ ] **Step 2: Prove Flask e2e is discriminating**

Temporarily set `os.environ["MONOCLE_TRACE_RETRIEVAL_KEY"] = "wrong"` at module top; re-run `test_flask_server_spans_returned`; it must FAIL on `raw is not None`. Restore. Paste both outputs in the report.

- [ ] **Step 3: aiohttp e2e**

Add an aiohttp server fixture (aiohttp.web app in a thread with its own event loop) + a test mirroring Step 1, asserting `answer_question` arrives via `_monocle_remote_spans`. Use `pytest.importorskip("aiohttp")`. If running an aiohttp server in-thread proves flaky under pytest, assert against a buffered `web.Response` route (the common path) and note streaming coverage is provided by the Task 5 unit test.

Run the file; expected PASS.

- [ ] **Step 4: Lambda + Azure wrapper-level e2e**

Add tests that import the route wrappers (`monocle_trace_lambda_function_route`, and the Azure equivalent), decorate a handler that emits a child span via `@monocle_trace_method`, invoke it with a fake event/`req` carrying `x-monocle-retrieve-traces: fw-s3cret`, and assert the returned dict / `HttpResponse` body+header carry the trailer and that a client strip recovers the span. `pytest.importorskip` azure-functions where needed.

Run the file; expected PASS (or SKIP for azure if not installed).

- [ ] **Step 5: Update the README**

In `README_HTTP_RUNNER.md`, update the "Known limitations" / server-support section: server-side span return now works for **FastAPI, Flask, aiohttp, AWS Lambda, Azure Functions**; streaming supported on FastAPI (ASGI), Flask (WSGI app_iter), and aiohttp (StreamResponse) — buffered for Lambda/Azure Functions (no streaming). Note any framework degraded to buffered-only if that happened in Task 3/5.

- [ ] **Step 6: Full sweep + commit**

Run:
```
pytest apptrace/tests/unit -k "trace_return or lambda_trace or azfunc_trace or flask_trace or aiohttp_trace" -v
pytest test_tools/tests/integration/test_http_trace_return_frameworks.py -v
```
Expected: all green (azure may SKIP).

```bash
git add test_tools/tests/integration/test_http_trace_return_frameworks.py test_tools/src/monocle_test_tools/runner/README_HTTP_RUNNER.md
git commit -m "test(test-tools): e2e trace-return across Flask/aiohttp/Lambda/Azure; docs"
```

---

## Self-Review Notes

- **Spec coverage:** shared helpers (Task 1); Lambda (2); Azure (3); Flask buffered+streaming (4); aiohttp buffered+streaming (5); e2e + docs (6). FastAPI unchanged (delegates via Task 1).
- **Uncertain internals carry investigation steps + fallbacks:** Azure `HttpResponse` private buffer (Task 3 Step 1 + response_processor fallback); aiohttp streaming hooks (Task 5 Step 3 + buffered-only fallback).
- **Deny/inert safety** is asserted per framework (noop-when-disabled tests) and end-to-end (discriminating Flask test).
- **No apptrace→test_tools import**; injection reuses the shared codec/exporter.
