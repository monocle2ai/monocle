# HTTP Span Piggyback — Plan 1 (core + requests client + test_tools + FastAPI)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return a FastAPI server's Monocle child spans in-band on the HTTP response (trailer block) so a local `HttpRunner` test can assert against them offline, with no Okahu round-trip.

**Architecture:** Server buffers only opted-in spans (scope-filtered in-memory exporter), appends `<delimiter><gzip+base64(json)>` after the real response body, and announces it via a tiny `x-monocle-traces` response header carrying a per-response delimiter. The `requests` client instrumentation strips the trailer and stashes raw span JSON on the response; the new `HttpRunner` deserializes it into the validator's memory exporter.

**Tech Stack:** Python, OpenTelemetry SDK, FastAPI/Starlette (ASGI), `requests`, pytest.

## Global Constraints

- Feature is inert unless server env `MONOCLE_ENABLE_TRACE_RETURN=true` **and** client sends request header `x-monocle-retrieve-traces: true`. Copy these exact strings.
- `apptrace` must NOT import from `test_tools` (dependency direction is one-way).
- Reuse existing `serialize_span()` in `apptrace/src/monocle_apptrace/exporters/base_exporter.py` for the span wire format (same format `JSONSpanLoader` reads).
- Scope name is exactly `monocle_trace_return`; span attribute is `scope.monocle_trace_return`.
- Response header is exactly `x-monocle-traces`, value format `v1; delim=<uuid4hex>`.
- Streaming responses must NOT be buffered whole; the trailer is appended as a final chunk.
- Run tests from repo root (unified `pytest.ini`).

---

### Task 1: Constants

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/common/constants.py`

**Interfaces:**
- Produces: `MONOCLE_TRACE_RETURN_ENABLED_ENV`, `TRACE_RETURN_REQUEST_HEADER`, `TRACE_RETURN_RESPONSE_HEADER`, `TRACE_RETURN_SCOPE_NAME`, `TRACE_RETURN_VERSION` (all `str`).

- [ ] **Step 1: Add constants**

Append to `constants.py`:

```python
# HTTP response span piggyback (trace return)
MONOCLE_TRACE_RETURN_ENABLED_ENV = "MONOCLE_ENABLE_TRACE_RETURN"
TRACE_RETURN_REQUEST_HEADER = "x-monocle-retrieve-traces"
TRACE_RETURN_RESPONSE_HEADER = "x-monocle-traces"
TRACE_RETURN_SCOPE_NAME = "monocle_trace_return"
TRACE_RETURN_VERSION = "v1"
```

- [ ] **Step 2: Verify import**

Run: `python -c "from monocle_apptrace.instrumentation.common.constants import MONOCLE_TRACE_RETURN_ENABLED_ENV, TRACE_RETURN_REQUEST_HEADER, TRACE_RETURN_RESPONSE_HEADER, TRACE_RETURN_SCOPE_NAME, TRACE_RETURN_VERSION; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/common/constants.py
git commit -m "feat(apptrace): add trace-return constants"
```

---

### Task 2: Trace-return codec module (pure functions)

**Files:**
- Create: `apptrace/src/monocle_apptrace/instrumentation/common/trace_return.py`
- Test: `apptrace/tests/unit/test_trace_return_codec.py`

**Interfaces:**
- Consumes: `serialize_span` from `monocle_apptrace.exporters.base_exporter`; constants from Task 1.
- Produces:
  - `is_trace_return_enabled() -> bool`
  - `is_trace_return_requested(headers: dict) -> bool`
  - `make_delimiter() -> str`
  - `encode_spans(spans: list) -> str` (base64 str of gzip of JSON list)
  - `decode_payload(payload: str) -> str` (raw JSON string of span-dict list)
  - `build_trailer_bytes(spans: list, delimiter: str) -> bytes`
  - `build_response_header_value(delimiter: str) -> str`
  - `parse_delimiter_from_header(header_value: str) -> str | None`
  - `split_body_and_trailer(body: bytes, delimiter: str) -> tuple[bytes, str | None]`

- [ ] **Step 1: Write the failing test**

Create `apptrace/tests/unit/test_trace_return_codec.py`:

```python
import os
from monocle_apptrace.instrumentation.common import trace_return as tr


def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    assert tr.is_trace_return_enabled() is False
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    assert tr.is_trace_return_enabled() is True
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "TRUE")
    assert tr.is_trace_return_enabled() is True
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "false")
    assert tr.is_trace_return_enabled() is False


def test_requested_checks_header():
    assert tr.is_trace_return_requested({"x-monocle-retrieve-traces": "true"}) is True
    assert tr.is_trace_return_requested({"X-Monocle-Retrieve-Traces": "TRUE"}) is True
    assert tr.is_trace_return_requested({"other": "true"}) is False
    assert tr.is_trace_return_requested({}) is False


def test_header_value_roundtrip():
    delim = tr.make_delimiter()
    value = tr.build_response_header_value(delim)
    assert value.startswith("v1;")
    assert tr.parse_delimiter_from_header(value) == delim
    assert tr.parse_delimiter_from_header("garbage") is None


def test_split_body_and_trailer():
    delim = "__MONOCLE_TR__abcd"
    clean = b'{"answer": "hi"}'
    trailer = delim.encode("utf-8") + b"PAYLOAD"
    body = clean + trailer
    got_clean, got_payload = tr.split_body_and_trailer(body, delim)
    assert got_clean == clean
    assert got_payload == "PAYLOAD"
    # no delimiter present -> payload None, body unchanged
    got_clean2, got_payload2 = tr.split_body_and_trailer(clean, delim)
    assert got_clean2 == clean
    assert got_payload2 is None
```

Run: `pytest apptrace/tests/unit/test_trace_return_codec.py -v`
Expected: FAIL — `ModuleNotFoundError: monocle_apptrace.instrumentation.common.trace_return`

- [ ] **Step 2: Implement the codec module**

Create `apptrace/src/monocle_apptrace/instrumentation/common/trace_return.py`:

```python
import base64
import gzip
import json
import logging
import os
import uuid

from monocle_apptrace.exporters.base_exporter import serialize_span
from monocle_apptrace.instrumentation.common.constants import (
    MONOCLE_TRACE_RETURN_ENABLED_ENV,
    TRACE_RETURN_REQUEST_HEADER,
    TRACE_RETURN_VERSION,
)

logger = logging.getLogger(__name__)

_DELIMITER_PREFIX = "\n__MONOCLE_TRACES__"


def is_trace_return_enabled() -> bool:
    return os.environ.get(MONOCLE_TRACE_RETURN_ENABLED_ENV, "false").lower() == "true"


def is_trace_return_requested(headers: dict) -> bool:
    if not headers:
        return False
    for key, value in headers.items():
        if str(key).lower() == TRACE_RETURN_REQUEST_HEADER and str(value).lower() == "true":
            return True
    return False


def make_delimiter() -> str:
    return f"{_DELIMITER_PREFIX}{uuid.uuid4().hex}__"


def encode_spans(spans: list) -> str:
    span_dicts = [serialize_span(span) for span in spans]
    raw = json.dumps(span_dicts).encode("utf-8")
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def decode_payload(payload: str) -> str:
    raw = gzip.decompress(base64.b64decode(payload.encode("ascii")))
    return raw.decode("utf-8")


def build_trailer_bytes(spans: list, delimiter: str) -> bytes:
    return delimiter.encode("utf-8") + encode_spans(spans).encode("ascii")


def build_response_header_value(delimiter: str) -> str:
    return f"{TRACE_RETURN_VERSION}; delim={delimiter}"


def parse_delimiter_from_header(header_value: str):
    if not header_value or "delim=" not in header_value:
        return None
    return header_value.split("delim=", 1)[1].strip()


def split_body_and_trailer(body: bytes, delimiter: str):
    marker = delimiter.encode("utf-8")
    idx = body.find(marker)
    if idx == -1:
        return body, None
    clean = body[:idx]
    payload = body[idx + len(marker):].decode("ascii")
    return clean, payload
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest apptrace/tests/unit/test_trace_return_codec.py -v`
Expected: PASS (4 tests)

- [ ] **Step 4: Add an end-to-end codec round-trip test**

Append to `apptrace/tests/unit/test_trace_return_codec.py`:

```python
def test_encode_decode_roundtrip():
    class FakeSpan:
        def to_json(self):
            return '{"name": "inference", "status": {"status_code": "OK"}}'
    delim = tr.make_delimiter()
    trailer = tr.build_trailer_bytes([FakeSpan(), FakeSpan()], delim)
    body = b'{"answer": "hi"}' + trailer
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert clean == b'{"answer": "hi"}'
    decoded = tr.decode_payload(payload)
    import json as _json
    spans = _json.loads(decoded)
    assert len(spans) == 2
    assert spans[0]["name"] == "inference"
```

Run: `pytest apptrace/tests/unit/test_trace_return_codec.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/common/trace_return.py apptrace/tests/unit/test_trace_return_codec.py
git commit -m "feat(apptrace): trace-return codec (encode/decode/split trailer)"
```

---

### Task 3: Scope-filtered in-memory exporter

**Files:**
- Create: `apptrace/src/monocle_apptrace/exporters/trace_return_exporter.py`
- Test: `apptrace/tests/unit/test_trace_return_exporter.py`

**Interfaces:**
- Consumes: `MonocleInMemorySpanExporter`; `TRACE_RETURN_SCOPE_NAME`.
- Produces:
  - `class TraceReturnSpanExporter(MonocleInMemorySpanExporter)` — `export()` keeps only spans with attribute `scope.monocle_trace_return`; method `pop_spans_for_trace(trace_id: int) -> list`.
  - `get_trace_return_exporter() -> TraceReturnSpanExporter` (module singleton)
  - `maybe_trace_return_processor()` -> `SimpleSpanProcessor | None` (a processor wrapping the singleton, only when `is_trace_return_enabled()`).

- [ ] **Step 1: Write the failing test**

Create `apptrace/tests/unit/test_trace_return_exporter.py`:

```python
from monocle_apptrace.exporters.trace_return_exporter import (
    TraceReturnSpanExporter,
    maybe_trace_return_processor,
)


class FakeCtx:
    def __init__(self, trace_id):
        self.trace_id = trace_id


class FakeSpan:
    def __init__(self, trace_id, tagged):
        attrs = {"monocle_apptrace.version": "1.0"}
        if tagged:
            attrs["scope.monocle_trace_return"] = "true"
        self._attributes = attrs
        self._ctx = FakeCtx(trace_id)

    @property
    def attributes(self):
        return self._attributes

    def get_span_context(self):
        return self._ctx


def test_export_keeps_only_tagged_spans():
    exp = TraceReturnSpanExporter()
    exp.export([FakeSpan(1, tagged=True), FakeSpan(1, tagged=False)])
    stored = exp.get_finished_spans()
    assert len(stored) == 1


def test_pop_spans_for_trace_evicts_by_trace_id():
    exp = TraceReturnSpanExporter()
    exp.export([FakeSpan(1, tagged=True), FakeSpan(2, tagged=True), FakeSpan(1, tagged=True)])
    popped = exp.pop_spans_for_trace(1)
    assert len(popped) == 2
    # trace 1 evicted, trace 2 remains
    remaining = exp.get_finished_spans()
    assert len(remaining) == 1
    assert remaining[0].get_span_context().trace_id == 2


def test_maybe_processor_gated_by_env(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    assert maybe_trace_return_processor() is None
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    assert maybe_trace_return_processor() is not None
```

Run: `pytest apptrace/tests/unit/test_trace_return_exporter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 2: Implement the exporter**

Create `apptrace/src/monocle_apptrace/exporters/trace_return_exporter.py`:

```python
import logging
import threading

from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from monocle_apptrace.exporters.base_exporter import MonocleInMemorySpanExporter
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_SCOPE_NAME

logger = logging.getLogger(__name__)

_SCOPE_ATTR = f"scope.{TRACE_RETURN_SCOPE_NAME}"


class TraceReturnSpanExporter(MonocleInMemorySpanExporter):
    """In-memory exporter that stores ONLY spans tagged with the trace-return scope."""

    def __init__(self):
        super().__init__()
        self._tr_lock = threading.Lock()

    def export(self, spans):
        tagged = [s for s in spans if s.attributes and s.attributes.get(_SCOPE_ATTR) is not None]
        if not tagged:
            return SpanExportResult.SUCCESS
        with self._tr_lock:
            return super().export(tagged)

    def pop_spans_for_trace(self, trace_id: int) -> list:
        """Return and evict all buffered spans whose trace_id matches."""
        with self._tr_lock:
            all_spans = list(self.get_finished_spans())
            matched = [s for s in all_spans if s.get_span_context().trace_id == trace_id]
            remaining = [s for s in all_spans if s.get_span_context().trace_id != trace_id]
            self.clear()
            if remaining:
                super().export(remaining)
        return matched


_trace_return_exporter = None
_singleton_lock = threading.Lock()


def get_trace_return_exporter() -> TraceReturnSpanExporter:
    global _trace_return_exporter
    with _singleton_lock:
        if _trace_return_exporter is None:
            _trace_return_exporter = TraceReturnSpanExporter()
        return _trace_return_exporter


def maybe_trace_return_processor():
    """Return a SimpleSpanProcessor around the singleton exporter, only if the feature is enabled."""
    from monocle_apptrace.instrumentation.common.trace_return import is_trace_return_enabled
    if not is_trace_return_enabled():
        return None
    return SimpleSpanProcessor(get_trace_return_exporter())
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest apptrace/tests/unit/test_trace_return_exporter.py -v`
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add apptrace/src/monocle_apptrace/exporters/trace_return_exporter.py apptrace/tests/unit/test_trace_return_exporter.py
git commit -m "feat(apptrace): scope-filtered TraceReturnSpanExporter"
```

---

### Task 4: Register the exporter in setup_monocle_telemetry

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/common/instrumentor.py:337` (span_processors assembly)
- Test: `apptrace/tests/unit/test_trace_return_setup.py`

**Interfaces:**
- Consumes: `maybe_trace_return_processor` (Task 3).
- Produces: when env enabled, the singleton `TraceReturnSpanExporter` receives spans emitted under the global provider.

- [ ] **Step 1: Write the failing test**

Create `apptrace/tests/unit/test_trace_return_setup.py`:

```python
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common import instrumentor as inst


def test_processor_added_when_enabled(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    # helper under test: appends the trace-return processor when enabled
    result = inst._append_trace_return_processor([])
    assert len(result) == 1
    assert isinstance(result[0], SimpleSpanProcessor)


def test_processor_absent_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    result = inst._append_trace_return_processor([])
    assert result == []
```

Run: `pytest apptrace/tests/unit/test_trace_return_setup.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_append_trace_return_processor'`

- [ ] **Step 2: Add the helper and call it in setup**

In `instrumentor.py`, add near the other module functions (e.g. after `get_monocle_span_processor`):

```python
def _append_trace_return_processor(span_processors):
    """Append the trace-return SimpleSpanProcessor when MONOCLE_ENABLE_TRACE_RETURN is on."""
    from monocle_apptrace.exporters.trace_return_exporter import maybe_trace_return_processor
    proc = maybe_trace_return_processor()
    if proc is not None:
        span_processors = list(span_processors) + [proc]
    return span_processors
```

Then modify the assembly at `instrumentor.py:337` from:

```python
    span_processors = span_processors or [BatchSpanProcessor(exporter) for exporter in exporters]
```

to:

```python
    span_processors = span_processors or [BatchSpanProcessor(exporter) for exporter in exporters]
    span_processors = _append_trace_return_processor(span_processors)
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest apptrace/tests/unit/test_trace_return_setup.py -v`
Expected: PASS (2 tests)

- [ ] **Step 4: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/common/instrumentor.py apptrace/tests/unit/test_trace_return_setup.py
git commit -m "feat(apptrace): register trace-return exporter in setup when enabled"
```

---

### Task 5: Set the trace-return scope in extract_http_headers

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/common/utils.py:367-376` (`extract_http_headers`)
- Test: `apptrace/tests/unit/test_trace_return_scope.py`

**Interfaces:**
- Consumes: `is_trace_return_enabled`, `is_trace_return_requested` (Task 2); `TRACE_RETURN_SCOPE_NAME`.
- Produces: when enabled + requested, `extract_http_headers()` sets `scope.monocle_trace_return` for the request context (visible via `get_scopes()`).

- [ ] **Step 1: Write the failing test**

Create `apptrace/tests/unit/test_trace_return_scope.py`:

```python
from opentelemetry.context import attach, detach
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, get_scopes, clear_http_scopes


def test_scope_set_when_enabled_and_requested(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    token = extract_http_headers({"x-monocle-retrieve-traces": "true"})
    try:
        assert "monocle_trace_return" in get_scopes()
    finally:
        clear_http_scopes(token)


def test_scope_absent_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    token = extract_http_headers({"x-monocle-retrieve-traces": "true"})
    try:
        assert "monocle_trace_return" not in get_scopes()
    finally:
        clear_http_scopes(token)


def test_scope_absent_when_not_requested(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    token = extract_http_headers({"other": "true"})
    try:
        assert "monocle_trace_return" not in get_scopes()
    finally:
        clear_http_scopes(token)
```

Run: `pytest apptrace/tests/unit/test_trace_return_scope.py -v`
Expected: FAIL — assertion error (scope not set)

- [ ] **Step 2: Modify extract_http_headers**

In `utils.py`, replace the body of `extract_http_headers` (currently lines 367-376) with:

```python
def extract_http_headers(headers) -> object:
    global http_scopes
    trace_context:Context = extract(headers, context=get_current())
    trace_context = set_value(ADD_NEW_WORKFLOW, True, trace_context)
    imported_scope:dict[str, object] = {}
    for http_header, http_scope in http_scopes.items():
        if http_header in headers:
            imported_scope[http_scope] = f"{http_header}: {headers[http_header]}"
    # trace-return opt-in: tag spans so the scope-filtered exporter captures them
    from monocle_apptrace.instrumentation.common.trace_return import (
        is_trace_return_enabled, is_trace_return_requested)
    from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_SCOPE_NAME
    if is_trace_return_enabled() and is_trace_return_requested(headers):
        imported_scope[TRACE_RETURN_SCOPE_NAME] = "true"
    token = set_scopes(imported_scope, trace_context)
    return token
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest apptrace/tests/unit/test_trace_return_scope.py -v`
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/common/utils.py apptrace/tests/unit/test_trace_return_scope.py
git commit -m "feat(apptrace): set trace-return scope on opted-in HTTP requests"
```

---

### Task 6: HttpSpanHandler shared trailer builder

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/common/span_handler.py:433` (`HttpSpanHandler`)
- Test: `apptrace/tests/unit/test_http_span_handler_trailer.py`

**Interfaces:**
- Consumes: `get_trace_return_exporter` (Task 3); trace_return codec (Task 2).
- Produces: `HttpSpanHandler.build_trace_return_trailer(trace_id: int, delimiter: str) -> bytes | None` — pops this trace's spans and returns trailer bytes, or `None` if none.

- [ ] **Step 1: Write the failing test**

Create `apptrace/tests/unit/test_http_span_handler_trailer.py`:

```python
from monocle_apptrace.instrumentation.common.span_handler import HttpSpanHandler
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


def test_build_trailer_returns_bytes_for_matching_trace():
    exp = get_trace_return_exporter()
    exp.clear()
    exp.export([FakeSpan(42)])
    delim = tr.make_delimiter()
    trailer = HttpSpanHandler().build_trace_return_trailer(42, delim)
    assert trailer is not None
    clean, payload = tr.split_body_and_trailer(b"BODY" + trailer, delim)
    assert clean == b"BODY"
    import json
    assert json.loads(tr.decode_payload(payload))[0]["name"] == "inference"


def test_build_trailer_none_when_no_spans():
    exp = get_trace_return_exporter()
    exp.clear()
    assert HttpSpanHandler().build_trace_return_trailer(99, tr.make_delimiter()) is None
```

Run: `pytest apptrace/tests/unit/test_http_span_handler_trailer.py -v`
Expected: FAIL — `AttributeError: 'HttpSpanHandler' object has no attribute 'build_trace_return_trailer'`

- [ ] **Step 2: Add the method to HttpSpanHandler**

In `span_handler.py`, inside `class HttpSpanHandler(SpanHandler):`, add:

```python
    def build_trace_return_trailer(self, trace_id: int, delimiter: str):
        """Pop this trace's captured spans and build the response trailer bytes.
        Returns None when there is nothing to return."""
        from monocle_apptrace.exporters.trace_return_exporter import get_trace_return_exporter
        from monocle_apptrace.instrumentation.common import trace_return as tr
        spans = get_trace_return_exporter().pop_spans_for_trace(trace_id)
        if not spans:
            return None
        return tr.build_trailer_bytes(spans, delimiter)
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest apptrace/tests/unit/test_http_span_handler_trailer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 4: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/common/span_handler.py apptrace/tests/unit/test_http_span_handler_trailer.py
git commit -m "feat(apptrace): HttpSpanHandler.build_trace_return_trailer"
```

---

### Task 7: FastAPI response injection (ASGI send wrapper)

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/metamodel/fastapi/_helper.py` (`_capture_response_body`, `fastapi_atask_wrapper`)
- Test: `apptrace/tests/unit/test_fastapi_trace_return_send.py`

**Interfaces:**
- Consumes: `HttpSpanHandler.build_trace_return_trailer` (Task 6); codec (Task 2).
- Produces: opted-in FastAPI responses carry the `x-monocle-traces` header and a trailer after the body; streaming and buffered both handled.

**Design of the send wrapper (`_inject_trace_return_send`):**
- Active only when `is_trace_return_enabled()` and the request opted in (header present in `scope['headers']`).
- Generates one `delimiter` up front.
- `http.response.start`: parse headers list; if a `content-length` header exists (buffered), HOLD the start message (store, don't forward) — we forward a corrected one at finalize. If no content-length (streaming/chunked), inject the `x-monocle-traces` header now and forward start immediately.
- `http.response.body`:
  - **Buffered** (start was held): accumulate `body`; when `more_body` is False, build trailer via `handler.build_trace_return_trailer(trace_id, delimiter)`; if trailer, recompute `content-length = len(body)+len(trailer)`, add `x-monocle-traces` header to the held start, forward start, then forward `{type: http.response.body, body: body+trailer}`. If no trailer, forward held start unchanged then original body.
  - **Streaming** (start already forwarded): forward each chunk as-is EXCEPT flip the final message's `more_body` to True and, after it, send one extra body message `{body: trailer, more_body: False}`. If trailer is None, forward the final chunk unchanged.
- `trace_id` is read at finalize via `opentelemetry.trace.get_current_span().get_span_context().trace_id`.

- [ ] **Step 1: Write the failing test**

Create `apptrace/tests/unit/test_fastapi_trace_return_send.py`. It drives the send wrapper directly with fake ASGI messages, with a stubbed handler so no real spans are needed:

```python
import asyncio
import pytest
from monocle_apptrace.instrumentation.metamodel.fastapi import _helper as h
from monocle_apptrace.instrumentation.common import trace_return as tr


class StubHandler:
    def __init__(self, trailer): self._trailer = trailer
    def build_trace_return_trailer(self, trace_id, delimiter): return self._trailer


def _run(coro): return asyncio.new_event_loop().run_until_complete(coro)


def _collect_send():
    sent = []
    async def send(message): sent.append(message)
    return sent, send


def test_buffered_injection_appends_trailer_and_fixes_length(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    delim = "\n__MONOCLE_TRACES__x__"
    trailer = delim.encode() + b"PAYLOAD"
    sent, send = _collect_send()
    scope = {"headers": [(b"x-monocle-retrieve-traces", b"true")]}
    wrapped = _run(h._inject_trace_return_send(scope, send, StubHandler(trailer), trace_id=7, delimiter=delim))
    _run(wrapped({"type": "http.response.start", "status": 200,
                  "headers": [(b"content-length", b"16"), (b"content-type", b"application/json")]}))
    _run(wrapped({"type": "http.response.body", "body": b'{"answer": "hi"}', "more_body": False}))
    start = [m for m in sent if m["type"] == "http.response.start"][0]
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    headers = dict(start["headers"])
    assert headers[b"content-length"] == str(16 + len(trailer)).encode()
    assert any(k == b"x-monocle-traces" for k, _ in start["headers"])
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert clean == b'{"answer": "hi"}'
    assert payload == "PAYLOAD"


def test_streaming_injection_appends_trailer_chunk(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    delim = "\n__MONOCLE_TRACES__y__"
    trailer = delim.encode() + b"PAYLOAD"
    sent, send = _collect_send()
    scope = {"headers": [(b"x-monocle-retrieve-traces", b"true")]}
    wrapped = _run(h._inject_trace_return_send(scope, send, StubHandler(trailer), trace_id=7, delimiter=delim))
    _run(wrapped({"type": "http.response.start", "status": 200,
                  "headers": [(b"content-type", b"text/event-stream")]}))
    _run(wrapped({"type": "http.response.body", "body": b"data: a\n\n", "more_body": True}))
    _run(wrapped({"type": "http.response.body", "body": b"data: b\n\n", "more_body": False}))
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert clean == b"data: a\n\ndata: b\n\n"
    assert payload == "PAYLOAD"
    # start forwarded with header, and stream not buffered (start before bodies)
    assert sent[0]["type"] == "http.response.start"


def test_no_injection_when_no_trailer(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    delim = "\n__MONOCLE_TRACES__z__"
    sent, send = _collect_send()
    scope = {"headers": [(b"x-monocle-retrieve-traces", b"true")]}
    wrapped = _run(h._inject_trace_return_send(scope, send, StubHandler(None), trace_id=7, delimiter=delim))
    _run(wrapped({"type": "http.response.start", "status": 200,
                  "headers": [(b"content-length", b"3")]}))
    _run(wrapped({"type": "http.response.body", "body": b"abc", "more_body": False}))
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert body == b"abc"
```

Run: `pytest apptrace/tests/unit/test_fastapi_trace_return_send.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_inject_trace_return_send'`

- [ ] **Step 2: Implement the send wrapper**

In `fastapi/_helper.py`, add imports at top:

```python
from monocle_apptrace.instrumentation.common.trace_return import (
    build_response_header_value,
    is_trace_return_enabled,
    is_trace_return_requested,
    make_delimiter,
)
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER
from opentelemetry import trace as _trace_api
```

Add the wrapper function:

```python
def _headers_from_scope(scope) -> dict:
    return {k.decode("utf-8").lower(): v.decode("utf-8") for k, v in scope.get("headers", [])}


async def _inject_trace_return_send(scope, send, handler, trace_id: int, delimiter: str):
    """Wrap ASGI send to append the trace-return trailer to the response.

    Buffered responses (content-length present): hold start, append trailer, fix length.
    Streaming responses (no content-length): forward start immediately, append a final
    trailer body chunk after the last real chunk.
    """
    header_bytes = (TRACE_RETURN_RESPONSE_HEADER.encode("ascii"),
                    build_response_header_value(delimiter).encode("ascii"))
    state = {"held_start": None, "buffered": False, "body_parts": []}

    async def _send(message):
        mtype = message.get("type")
        if mtype == "http.response.start":
            headers = list(message.get("headers", []))
            has_length = any(k.lower() == b"content-length" for k, _ in headers)
            if has_length:
                state["buffered"] = True
                state["held_start"] = message
                return  # defer until finalize
            headers.append(header_bytes)
            message = {**message, "headers": headers}
            await send(message)
            return

        if mtype == "http.response.body":
            body = message.get("body", b"")
            more = message.get("more_body", False)
            if state["buffered"]:
                state["body_parts"].append(body)
                if more:
                    return
                full_body = b"".join(state["body_parts"])
                trailer = handler.build_trace_return_trailer(trace_id, delimiter)
                start = state["held_start"]
                headers = list(start.get("headers", []))
                if trailer:
                    new_headers = []
                    for k, v in headers:
                        if k.lower() == b"content-length":
                            new_headers.append((k, str(len(full_body) + len(trailer)).encode("ascii")))
                        else:
                            new_headers.append((k, v))
                    new_headers.append(header_bytes)
                    await send({**start, "headers": new_headers})
                    await send({"type": "http.response.body", "body": full_body + trailer, "more_body": False})
                else:
                    await send(start)
                    await send({"type": "http.response.body", "body": full_body, "more_body": False})
                return
            # streaming path (start already forwarded)
            if more:
                await send(message)
                return
            trailer = handler.build_trace_return_trailer(trace_id, delimiter)
            if trailer:
                await send({**message, "body": body, "more_body": True})
                await send({"type": "http.response.body", "body": trailer, "more_body": False})
            else:
                await send(message)
            return

        await send(message)

    return _send
```

Now wire it into `fastapi_atask_wrapper` — replace its body so the trace-return send wrapper is layered when opted in. Change the section that wraps `send`:

```python
    scope, receive, send = args[0], args[1], args[2]

    if scope.get('method', 'GET') in ('POST', 'PUT', 'PATCH'):
        receive = await _buffer_request_body(scope, receive)

    send = await _capture_response_body(scope, send)

    # trace-return injection (opt-in)
    if is_trace_return_enabled() and is_trace_return_requested(_headers_from_scope(scope)):
        delimiter = make_delimiter()
        trace_id = _trace_api.get_current_span().get_span_context().trace_id
        send = await _inject_trace_return_send(scope, send, HttpSpanHandler(), trace_id, delimiter)

    args = (scope, receive, send) + args[3:]

    return await amonocle_wrapper(
        tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs
    )
```

Note: `HttpSpanHandler` is already imported in this file (line 11). `trace_id` read here is 0 if no span is active; the integration test (Task 9) confirms it is populated during a real request. If it is 0, `build_trace_return_trailer` will simply find no matching spans and skip — safe.

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest apptrace/tests/unit/test_fastapi_trace_return_send.py -v`
Expected: PASS (3 tests)

- [ ] **Step 4: Run existing FastAPI tests to check no regression**

Run: `pytest apptrace/tests -k fastapi -v`
Expected: PASS (or same pass/skip set as before this task)

- [ ] **Step 5: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/metamodel/fastapi/_helper.py apptrace/tests/unit/test_fastapi_trace_return_send.py
git commit -m "feat(apptrace): FastAPI response trailer injection for trace return"
```

---

### Task 8: requests client — strip trailer & stash spans

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/metamodel/requests/_helper.py` (`RequestSpanHandler`)
- Test: `apptrace/tests/unit/test_requests_trace_return_strip.py`

**Interfaces:**
- Consumes: codec (Task 2); `TRACE_RETURN_RESPONSE_HEADER`.
- Produces: after an instrumented `requests` call, if the response carried the trailer, `response._monocle_remote_spans` holds the raw span-JSON string and `response.content`/`.text` return the clean body.

- [ ] **Step 1: Write the failing test**

Create `apptrace/tests/unit/test_requests_trace_return_strip.py`:

```python
from monocle_apptrace.instrumentation.metamodel.requests._helper import RequestSpanHandler
from monocle_apptrace.instrumentation.common import trace_return as tr


class FakeResponse:
    def __init__(self, content, headers):
        self._content = content
        self.headers = headers


def test_strip_and_stash():
    delim = tr.make_delimiter()

    class FakeSpan:
        def to_json(self): return '{"name": "inference"}'
    trailer = tr.build_trailer_bytes([FakeSpan()], delim)
    resp = FakeResponse(b'{"answer": "hi"}' + trailer,
                        {"x-monocle-traces": tr.build_response_header_value(delim)})
    RequestSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=resp, ex=None, span=None, parent_span=None)
    assert resp._content == b'{"answer": "hi"}'
    import json
    spans = json.loads(resp._monocle_remote_spans)
    assert spans[0]["name"] == "inference"


def test_no_header_is_noop():
    resp = FakeResponse(b'{"answer": "hi"}', {})
    RequestSpanHandler().post_task_processing(
        to_wrap={}, wrapped=None, instance=None, args=(), kwargs={},
        result=resp, ex=None, span=None, parent_span=None)
    assert resp._content == b'{"answer": "hi"}'
    assert not hasattr(resp, "_monocle_remote_spans")
```

Run: `pytest apptrace/tests/unit/test_requests_trace_return_strip.py -v`
Expected: FAIL — `RequestSpanHandler` has no `post_task_processing` override (base does nothing with the response) → `_monocle_remote_spans` missing / content not stripped

- [ ] **Step 2: Add post_task_processing to RequestSpanHandler**

In `requests/_helper.py`, add imports:

```python
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER
from monocle_apptrace.instrumentation.common import trace_return as tr
```

Add to `class RequestSpanHandler(SpanHandler):`:

```python
    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        try:
            headers = getattr(result, "headers", None)
            header_value = headers.get(TRACE_RETURN_RESPONSE_HEADER) if headers else None
            if header_value:
                delimiter = tr.parse_delimiter_from_header(header_value)
                if delimiter:
                    body = getattr(result, "_content", None)
                    if isinstance(body, (bytes, bytearray)):
                        clean, payload = tr.split_body_and_trailer(bytes(body), delimiter)
                        if payload is not None:
                            result._content = clean
                            result._monocle_remote_spans = tr.decode_payload(payload)
        except Exception as e:
            logger.debug(f"trace-return strip failed: {e}")
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)
```

Add `import logging` + `logger = logging.getLogger(__name__)` at the top of the file if not already present.

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest apptrace/tests/unit/test_requests_trace_return_strip.py -v`
Expected: PASS (2 tests)

- [ ] **Step 4: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/metamodel/requests/_helper.py apptrace/tests/unit/test_requests_trace_return_strip.py
git commit -m "feat(apptrace): requests client strips trailer and stashes remote spans"
```

---

### Task 9: test_tools — rename runner, new HttpRunner, deserialize into validator

**Files:**
- Modify: `test_tools/src/monocle_test_tools/runner/agent_runner.py` (add `get_remote_spans`)
- Modify: `test_tools/src/monocle_test_tools/runner/runner.py` (rename mapping + add `HTTP`)
- Modify: `test_tools/src/monocle_test_tools/runner/http_runner.py` (rename class + new `HttpRunner`)
- Modify: `test_tools/src/monocle_test_tools/file_span_loader.py` (add `from_json_str`)
- Modify: `test_tools/src/monocle_test_tools/validator.py:275-299` (ingest remote spans)
- Test: `test_tools/tests/unit/test_http_runner_ingest.py`

**Interfaces:**
- Consumes: `response._monocle_remote_spans` (Task 8); `JSONSpanLoader` (extended).
- Produces:
  - `AgentRunner.get_remote_spans(self) -> list` (default `[]`).
  - `HttpOkahuRunner` (was `HttpRunner`) under `AgentTypes.HTTP_WITH_OKAHU = "http_with_okahu"`.
  - `HttpRunner` under new `AgentTypes.HTTP = "http"`; deserializes stashed spans; `get_remote_traces_source()` returns `None`.
  - `JSONSpanLoader.from_json_str(json_str: str) -> List[ReadableSpan]`.
  - Validator `run_agent[_async]` calls `add_remote_spans(runner.get_remote_spans())`.

- [ ] **Step 1: Write the failing test**

Create `test_tools/tests/unit/test_http_runner_ingest.py`:

```python
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
```

Run: `pytest test_tools/tests/unit/test_http_runner_ingest.py -v`
Expected: FAIL — `ImportError: cannot import name 'HttpOkahuRunner'`

- [ ] **Step 2: Add `from_json_str` to JSONSpanLoader**

In `file_span_loader.py`, add to `class JSONSpanLoader`:

```python
    @staticmethod
    def from_json_str(json_str: str) -> List[ReadableSpan]:
        """Load spans from a JSON string (list of span dicts in file/Okahu format)."""
        span_data = json.loads(json_str)
        return [JSONSpanLoader._from_dict(item) for item in span_data]
```

- [ ] **Step 3: Add `get_remote_spans` to AgentRunner**

In `agent_runner.py`, add to `class AgentRunner`:

```python
    def get_remote_spans(self) -> list:
        """Spans the runner obtained out-of-band (e.g. piggybacked on an HTTP response).
        Default: none."""
        return []
```

- [ ] **Step 4: Rename runner class + add new HttpRunner**

Replace `test_tools/src/monocle_test_tools/runner/http_runner.py` with:

```python
import asyncio
import logging
from typing import Any
import requests
from monocle_test_tools.runner.agent_runner import AgentRunner
from monocle_test_tools.file_span_loader import JSONSpanLoader
from monocle_apptrace.instrumentation.metamodel.requests._helper import RequestSpanHandler

logger = logging.getLogger(__name__)


class _BaseHttpRunner(AgentRunner):
    async def run_agent_async(self, root_agent: str, *args, **kwargs) -> Any:
        if root_agent is None or not isinstance(root_agent, str):
            raise ValueError("For HttpRunner, root_agent must be the target URL string.")
        try:
            RequestSpanHandler.set_trace_all_urls_for_test(True)
            kwargs["url"] = root_agent
            response = requests.request(**kwargs)
            logger.debug(f"HTTP response status={response.status_code}")
            response.raise_for_status()
            return response
        finally:
            RequestSpanHandler.set_trace_all_urls_for_test(False)

    def run_agent(self, root_agent: str, *args, **kwargs) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.run_agent_async(root_agent, *args, **kwargs))
                return future.result()
        return asyncio.run(self.run_agent_async(root_agent, *args, **kwargs))


class HttpOkahuRunner(_BaseHttpRunner):
    """HTTP runner that fetches server-side traces from Okahu (legacy behavior)."""

    def get_remote_traces_source(self) -> str:
        return "okahu"


class HttpRunner(_BaseHttpRunner):
    """HTTP runner that reads server-side spans piggybacked on the HTTP response."""

    def __init__(self):
        self._remote_spans = []

    async def run_agent_async(self, root_agent: str, *args, **kwargs) -> Any:
        response = await super().run_agent_async(root_agent, *args, **kwargs)
        self._capture_remote_spans(response)
        return response

    def _capture_remote_spans(self, response) -> None:
        raw = getattr(response, "_monocle_remote_spans", None)
        if raw:
            try:
                self._remote_spans = JSONSpanLoader.from_json_str(raw)
            except Exception as e:
                logger.warning(f"Failed to deserialize piggybacked spans: {e}")
                self._remote_spans = []

    def get_remote_spans(self) -> list:
        return self._remote_spans

    def get_remote_traces_source(self):
        return None
```

- [ ] **Step 5: Update the agent-type registry**

In `runner.py`, change the enum and mapping:

```python
class AgentTypes(str, Enum):
    GOOGLE_ADK = "google_adk"
    OPENAI = "openai"
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    LLAMAINDEX = "llamaindex"
    STRANDS = "strands"
    MSAGENT = "msagent"
    HTTP = "http"
    HTTP_WITH_OKAHU = "http_with_okahu"
```

And replace the HTTP branch of `get_agent_runner`:

```python
    elif runner_type == AgentTypes.HTTP:
        from .http_runner import HttpRunner
        return HttpRunner()
    elif runner_type == AgentTypes.HTTP_WITH_OKAHU:
        from .http_runner import HttpOkahuRunner
        return HttpOkahuRunner()
```

- [ ] **Step 6: Ingest remote spans in the validator**

In `validator.py`, in BOTH `run_agent` (after line 280 `self._fetch_remote_traces()`) and `run_agent_async` (after line 298), add:

```python
        remote_spans = agent_runner.get_remote_spans()
        if remote_spans:
            self.add_remote_spans(remote_spans)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest test_tools/tests/unit/test_http_runner_ingest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 8: Commit**

```bash
git add test_tools/src/monocle_test_tools/runner/agent_runner.py test_tools/src/monocle_test_tools/runner/runner.py test_tools/src/monocle_test_tools/runner/http_runner.py test_tools/src/monocle_test_tools/file_span_loader.py test_tools/src/monocle_test_tools/validator.py test_tools/tests/unit/test_http_runner_ingest.py
git commit -m "feat(test-tools): HttpRunner ingests piggybacked spans; rename legacy to HttpOkahuRunner"
```

---

### Task 10: End-to-end integration test (FastAPI + HttpRunner, no Okahu)

**Files:**
- Test: `test_tools/tests/integration/test_http_trace_return.py`

**Interfaces:**
- Consumes: everything above.
- Produces: proof that a locally-served FastAPI app returns its child spans in-band to `HttpRunner`, and that the feature is inert when disabled.

- [ ] **Step 1: Write the integration test**

Create `test_tools/tests/integration/test_http_trace_return.py`. It starts a FastAPI app in a background uvicorn thread, instruments Monocle with the env flag on, drives it with the fluent asserter, and checks the server's spans arrived locally:

```python
import os
import threading
import time
import socket
import pytest

pytest_plugins = ["monocle_test_tools.pytest_plugin"]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server():
    import uvicorn
    from fastapi import FastAPI
    from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method
    os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "true"
    app = FastAPI()

    @monocle_trace_method(span_name="answer_question")
    def _answer(q: str) -> str:
        return f"echo: {q}"

    @app.post("/chat")
    async def chat(payload: dict):
        # produces at least one child span under the fastapi.request span
        return {"answer": _answer(payload.get("q", ""))}

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


def test_server_spans_returned_in_band(server):
    """The FastAPI server's spans are piggybacked on the response and land in the validator."""
    from monocle_test_tools.validator import MonocleValidator
    validator = MonocleValidator()
    response = validator.run_agent(
        server + "/chat", "http",
        method="POST", json={"q": "hi"},
        headers={"x-monocle-retrieve-traces": "true"},
    )
    assert response.json()["answer"] == "echo: hi"   # body is clean (trailer stripped)
    # the server-side child span emitted inside the route arrived locally (no okahu)
    span_names = [s.name for s in validator.spans]
    assert "answer_question" in span_names
```

> Implementer note: the assertion that matters is that **server-generated spans
> arrive in `validator.spans` without any Okahu fetch**. The `@monocle_trace_method`
> decorator (imported from `monocle_apptrace.instrumentation.common.instrumentor`)
> guarantees a concrete `answer_question` child span under the `fastapi.request`
> span. If instrumentation ordering means the decorator must be applied after
> `setup_monocle_telemetry`, move the `_answer` definition accordingly.

- [ ] **Step 2: Run the integration test**

Run: `pytest test_tools/tests/integration/test_http_trace_return.py -v`
Expected: PASS — `answer_question` appears in `validator.spans`.

- [ ] **Step 3: Add the inert-when-disabled test**

Append to the file:

```python
def test_inert_when_disabled(server):
    os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "false"
    import requests
    r = requests.post(server + "/chat", json={"q": "x"},
                      headers={"x-monocle-retrieve-traces": "true"})
    assert "x-monocle-traces" not in {k.lower() for k in r.headers.keys()}
    assert r.json()["answer"] == "echo: x"
    os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "true"
```

Run: `pytest test_tools/tests/integration/test_http_trace_return.py -v`
Expected: PASS (2 tests)

- [ ] **Step 4: Full regression sweep**

Run: `pytest apptrace/tests/unit -v && pytest test_tools/tests -k "http or trace_return" -v`
Expected: PASS across the trace-return unit + integration suite; no new failures in the broader apptrace unit tests.

- [ ] **Step 5: Commit**

```bash
git add test_tools/tests/integration/test_http_trace_return.py
git commit -m "test(test-tools): e2e FastAPI trace-return via HttpRunner (no okahu)"
```

---

## Self-Review Notes

- **Spec coverage:** activation env+header (Tasks 1,5,7), scope-filtered exporter (Task 3), setup registration (Task 4), shared trailer builder on HttpSpanHandler (Task 6), trailer transport with streaming + Content-Length (Task 7), client strip/stash (Task 8), runner rename + new HttpRunner + ingest (Task 9), e2e + inert (Task 10). File-exporter wire format reused via `serialize_span`/`JSONSpanLoader` (Tasks 2,9).
- **Known-limitation alignment:** root HTTP span is not popped (it ends after send) — trailer contains only completed child spans, matching the spec.
- **Out of scope (Plan 2):** flask/aiohttp/lambda/azfunc response wiring — each reuses Tasks 1–6 + 8–9 unchanged and only adds its own equivalent of Task 7.
