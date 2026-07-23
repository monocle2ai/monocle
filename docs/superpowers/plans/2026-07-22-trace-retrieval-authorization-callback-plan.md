# Trace-Retrieval Authorization Callback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate HTTP trace retrieval with a pluggable per-request authorization callback (`MONOCLE_TRACE_RETRIEVAL_CALLBACK`) plus a secure default callback that compares the `x-monocle-retrieve-traces` header value to a server key (`MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY`); `HttpRunner` auto-injects the key from `MONOCLE_TRACE_RETRIEVAL_KEY`.

**Architecture:** Replace the per-request `is_trace_return_requested(headers)` (header == "true") with `is_trace_return_authorized(headers)` in the two existing gate sites, while keeping `MONOCLE_ENABLE_TRACE_RETURN` as the master switch. New logic lives in the existing `trace_return.py`.

**Tech Stack:** Python, importlib, hmac; OpenTelemetry; FastAPI; requests; pytest.

## Global Constraints

- Env var names, exact: `MONOCLE_TRACE_RETRIEVAL_CALLBACK`, `MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY`, `MONOCLE_TRACE_RETRIEVAL_KEY`.
- Callback spec format: `"pkg.module:callable"`. Any resolve failure (bad format, unimportable module, missing attr, not callable) → deny (`False`) + warning log.
- Callback raising at call time → deny (`False`) + warning log.
- Default callback denies unless `MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY` is set AND the case-insensitive `x-monocle-retrieve-traces` header value matches; compare with `hmac.compare_digest`.
- `MONOCLE_ENABLE_TRACE_RETURN=true` remains the master switch (unchanged); traces returned iff enabled AND authorized.
- `is_trace_return_requested` is removed once its call sites are swapped.
- Client key injection lives ONLY in `HttpRunner`, never in the generic `requests` instrumentation. Caller-supplied header wins.
- apptrace must NOT import from test_tools.
- TDD: failing test first. Run all tests from the repo root (shared `pytest.ini`).

---

### Task 1: Add authorization env-var constants

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/common/constants.py` (after line 268, the trace-return block)

**Interfaces:**
- Produces: `MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV`, `MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY_ENV`, `MONOCLE_TRACE_RETRIEVAL_KEY_ENV` (all `str`).

- [ ] **Step 1: Add constants**

After the existing `TRACE_RETURN_VERSION = "v1"` line, append:

```python
# Trace-retrieval authorization
MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV = "MONOCLE_TRACE_RETRIEVAL_CALLBACK"
MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY_ENV = "MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY"
MONOCLE_TRACE_RETRIEVAL_KEY_ENV = "MONOCLE_TRACE_RETRIEVAL_KEY"
```

- [ ] **Step 2: Verify import**

Run: `python -c "from monocle_apptrace.instrumentation.common.constants import MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV, MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY_ENV, MONOCLE_TRACE_RETRIEVAL_KEY_ENV; print('ok')"`
Expected: `ok` (if a stale site-packages copy shadows the source, this may fail — that's fine; the pytest runs below use the working-tree source. If so, verify by importing the file directly or skip to Task 2's tests.)

- [ ] **Step 3: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/common/constants.py
git commit -m "feat(apptrace): add trace-retrieval authorization env constants"
```

---

### Task 2: Default callback + authorization dispatch (additive)

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/common/trace_return.py`
- Test: `apptrace/tests/unit/test_trace_return_authz.py`

**Interfaces:**
- Consumes: constants from Task 1; `TRACE_RETURN_REQUEST_HEADER`.
- Produces:
  - `default_trace_retrieval_callback(headers: dict) -> bool`
  - `is_trace_return_authorized(headers: dict) -> bool`

(Do NOT remove `is_trace_return_requested` in this task — that happens in Task 3 after call sites are swapped. This task is purely additive.)

- [ ] **Step 1: Write the failing tests**

Create `apptrace/tests/unit/test_trace_return_authz.py`:

```python
from monocle_apptrace.instrumentation.common import trace_return as tr


# ---- default callback ----

def test_default_callback_match(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    assert tr.default_trace_retrieval_callback({"x-monocle-retrieve-traces": "s3cret"}) is True


def test_default_callback_mismatch(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    assert tr.default_trace_retrieval_callback({"x-monocle-retrieve-traces": "wrong"}) is False


def test_default_callback_header_absent(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    assert tr.default_trace_retrieval_callback({"other": "s3cret"}) is False


def test_default_callback_key_unset(monkeypatch):
    monkeypatch.delenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", raising=False)
    assert tr.default_trace_retrieval_callback({"x-monocle-retrieve-traces": "anything"}) is False


def test_default_callback_case_insensitive_header(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    assert tr.default_trace_retrieval_callback({"X-Monocle-Retrieve-Traces": "s3cret"}) is True


# ---- dispatch ----

def test_authorized_uses_default_when_no_custom(monkeypatch):
    monkeypatch.delenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", raising=False)
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    assert tr.is_trace_return_authorized({"x-monocle-retrieve-traces": "s3cret"}) is True
    assert tr.is_trace_return_authorized({"x-monocle-retrieve-traces": "no"}) is False


def test_authorized_custom_callback(tmp_path, monkeypatch):
    mod = tmp_path / "cb_ok.py"
    mod.write_text("def cb(headers):\n    return headers.get('x-ok') == 'yes'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "cb_ok:cb")
    assert tr.is_trace_return_authorized({"x-ok": "yes"}) is True
    assert tr.is_trace_return_authorized({"x-ok": "no"}) is False


def test_authorized_unresolvable_callback_denies(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "no_such_module_xyz:cb")
    assert tr.is_trace_return_authorized({"x-monocle-retrieve-traces": "x"}) is False


def test_authorized_not_callable_denies(tmp_path, monkeypatch):
    mod = tmp_path / "cb_notcallable.py"
    mod.write_text("NOT_A_FUNC = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "cb_notcallable:NOT_A_FUNC")
    assert tr.is_trace_return_authorized({}) is False


def test_authorized_callback_raises_denies(tmp_path, monkeypatch):
    mod = tmp_path / "cb_boom.py"
    mod.write_text("def cb(headers):\n    raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "cb_boom:cb")
    assert tr.is_trace_return_authorized({}) is False


def test_authorized_malformed_spec_denies(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "no_colon_here")
    assert tr.is_trace_return_authorized({}) is False
```

Run: `pytest apptrace/tests/unit/test_trace_return_authz.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'default_trace_retrieval_callback'`

- [ ] **Step 2: Implement the functions**

In `apptrace/src/monocle_apptrace/instrumentation/common/trace_return.py`, add these imports at the top (alongside the existing ones):

```python
import hmac
import importlib
import logging
```

Add to the constants import block (it currently imports `MONOCLE_TRACE_RETURN_ENABLED_ENV, TRACE_RETURN_REQUEST_HEADER, TRACE_RETURN_VERSION`):

```python
from monocle_apptrace.instrumentation.common.constants import (
    MONOCLE_TRACE_RETURN_ENABLED_ENV,
    MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV,
    MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY_ENV,
    TRACE_RETURN_REQUEST_HEADER,
    TRACE_RETURN_VERSION,
)
```

Add a module logger after the imports:

```python
logger = logging.getLogger(__name__)
```

Add these functions (place them after `is_trace_return_requested`):

```python
def _get_header_case_insensitive(headers: dict, name: str):
    if not headers:
        return None
    lname = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lname:
            return value
    return None


def default_trace_retrieval_callback(headers: dict) -> bool:
    """Default authorization: the request's x-monocle-retrieve-traces header
    value must equal the server key MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY."""
    key = os.environ.get(MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY_ENV)
    if not key:
        return False
    value = _get_header_case_insensitive(headers, TRACE_RETURN_REQUEST_HEADER)
    if value is None:
        return False
    return hmac.compare_digest(str(value), str(key))


def _resolve_callback(spec: str):
    """Resolve a 'pkg.module:callable' spec to a callable, or None on failure."""
    if ":" not in spec:
        logger.warning("Invalid MONOCLE_TRACE_RETRIEVAL_CALLBACK spec (expected 'module:callable'): %s", spec)
        return None
    module_path, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_path)
        candidate = getattr(module, attr)
    except (ImportError, AttributeError) as e:
        logger.warning("Could not load trace-retrieval callback '%s': %s", spec, e)
        return None
    if not callable(candidate):
        logger.warning("Trace-retrieval callback '%s' is not callable", spec)
        return None
    return candidate


def is_trace_return_authorized(headers: dict) -> bool:
    """Per-request authorization gate for trace retrieval. Uses the callback
    configured via MONOCLE_TRACE_RETRIEVAL_CALLBACK, or the default callback.
    Any failure to load or run the callback denies (returns False)."""
    spec = os.environ.get(MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV)
    if spec:
        callback = _resolve_callback(spec)
        if callback is None:
            return False
    else:
        callback = default_trace_retrieval_callback
    try:
        return bool(callback(headers))
    except Exception as e:
        logger.warning("Trace-retrieval authorization callback raised: %s", e)
        return False
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest apptrace/tests/unit/test_trace_return_authz.py -v`
Expected: PASS (11 tests)

- [ ] **Step 4: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/common/trace_return.py apptrace/tests/unit/test_trace_return_authz.py
git commit -m "feat(apptrace): trace-retrieval authorization callback + secure default"
```

---

### Task 3: Swap the gate sites to authorization; remove is_trace_return_requested

**Files:**
- Modify: `apptrace/src/monocle_apptrace/instrumentation/common/utils.py` (extract_http_headers, ~line 375-380)
- Modify: `apptrace/src/monocle_apptrace/instrumentation/metamodel/fastapi/_helper.py` (import line ~15; gate at ~line 473)
- Modify: `apptrace/src/monocle_apptrace/instrumentation/common/trace_return.py` (remove `is_trace_return_requested`)
- Modify: `apptrace/tests/unit/test_trace_return_scope.py` (rewrite for the key-based gate)

**Interfaces:**
- Consumes: `is_trace_return_authorized` (Task 2).
- Produces: both gate sites now use `is_trace_return_enabled() and is_trace_return_authorized(headers)`. `is_trace_return_requested` no longer exists.

- [ ] **Step 1: Update the extract_http_headers test first (RED)**

Replace the body of `apptrace/tests/unit/test_trace_return_scope.py` with (the gate is now key-based, not "true"):

```python
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, get_scopes, clear_http_scopes


def test_scope_set_when_enabled_and_authorized(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    monkeypatch.delenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", raising=False)
    token = extract_http_headers({"x-monocle-retrieve-traces": "s3cret"})
    try:
        assert "monocle_trace_return" in get_scopes()
    finally:
        clear_http_scopes(token)


def test_scope_absent_when_disabled(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    token = extract_http_headers({"x-monocle-retrieve-traces": "s3cret"})
    try:
        assert "monocle_trace_return" not in get_scopes()
    finally:
        clear_http_scopes(token)


def test_scope_absent_when_not_authorized(monkeypatch):
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    monkeypatch.delenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", raising=False)
    token = extract_http_headers({"x-monocle-retrieve-traces": "wrong-key"})
    try:
        assert "monocle_trace_return" not in get_scopes()
    finally:
        clear_http_scopes(token)
```

Run: `pytest apptrace/tests/unit/test_trace_return_scope.py -v`
Expected: FAIL — `test_scope_set_when_enabled_and_authorized` fails (extract_http_headers still uses `is_trace_return_requested`, which checks for value "true", so "s3cret" is not recognized).

- [ ] **Step 2: Swap the extract_http_headers gate**

In `utils.py`, change the trace-return block from:

```python
    from monocle_apptrace.instrumentation.common.trace_return import (
        is_trace_return_enabled, is_trace_return_requested)
    from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_SCOPE_NAME
    if is_trace_return_enabled() and is_trace_return_requested(headers):
        imported_scope[TRACE_RETURN_SCOPE_NAME] = "true"
```

to:

```python
    from monocle_apptrace.instrumentation.common.trace_return import (
        is_trace_return_enabled, is_trace_return_authorized)
    from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_SCOPE_NAME
    if is_trace_return_enabled() and is_trace_return_authorized(headers):
        imported_scope[TRACE_RETURN_SCOPE_NAME] = "true"
```

- [ ] **Step 3: Swap the FastAPI wiring gate**

In `fastapi/_helper.py`, update the import (change `is_trace_return_requested` → `is_trace_return_authorized`):

```python
from monocle_apptrace.instrumentation.common.trace_return import (
    build_response_header_value,
    is_trace_return_authorized,
    is_trace_return_enabled,
    make_delimiter,
)
```

(Keep whatever other names are already imported from that module; only replace `is_trace_return_requested` with `is_trace_return_authorized`.)

Then change the gate line from:

```python
    if is_trace_return_enabled() and is_trace_return_requested(_headers_from_scope(scope)):
```

to:

```python
    if is_trace_return_enabled() and is_trace_return_authorized(_headers_from_scope(scope)):
```

- [ ] **Step 4: Remove is_trace_return_requested**

In `trace_return.py`, delete the `is_trace_return_requested` function (lines 21-27 in the current file). Confirm no remaining references:

Run: `grep -rn "is_trace_return_requested" apptrace/src apptrace/tests test_tools`
Expected: no output (all references removed).

- [ ] **Step 5: Run tests to verify GREEN + no regression**

Run:
```
pytest apptrace/tests/unit/test_trace_return_scope.py -v
pytest apptrace/tests/unit/test_trace_return_authz.py -v
pytest apptrace/tests -k fastapi -v
```
Expected: scope tests PASS (3), authz tests PASS (11), fastapi tests show no new failures vs. before (report counts; pre-existing collection errors for missing optional deps are unrelated).

- [ ] **Step 6: Commit**

```bash
git add apptrace/src/monocle_apptrace/instrumentation/common/utils.py apptrace/src/monocle_apptrace/instrumentation/metamodel/fastapi/_helper.py apptrace/src/monocle_apptrace/instrumentation/common/trace_return.py apptrace/tests/unit/test_trace_return_scope.py
git commit -m "feat(apptrace): gate trace retrieval via authorization callback"
```

---

### Task 4: HttpRunner auto-injects the retrieval key

**Files:**
- Modify: `test_tools/src/monocle_test_tools/runner/http_runner.py`
- Test: `test_tools/tests/unit/test_http_runner_key_injection.py`

**Interfaces:**
- Consumes: `MONOCLE_TRACE_RETRIEVAL_KEY_ENV`, `TRACE_RETURN_REQUEST_HEADER` (from apptrace constants).
- Produces: `HttpRunner` adds `x-monocle-retrieve-traces: <key>` to request headers when `MONOCLE_TRACE_RETRIEVAL_KEY` is set and the caller did not already set that header.

- [ ] **Step 1: Write the failing test**

Create `test_tools/tests/unit/test_http_runner_key_injection.py`:

```python
from monocle_test_tools.runner.http_runner import HttpRunner


def test_injects_key_when_env_set(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_KEY", "s3cret")
    runner = HttpRunner()
    kwargs = {"method": "POST"}
    runner._maybe_inject_retrieval_key(kwargs)
    assert kwargs["headers"]["x-monocle-retrieve-traces"] == "s3cret"


def test_caller_header_wins(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_KEY", "s3cret")
    runner = HttpRunner()
    kwargs = {"headers": {"X-Monocle-Retrieve-Traces": "caller-value"}}
    runner._maybe_inject_retrieval_key(kwargs)
    # caller-supplied value preserved (case-insensitive detection), key not overwritten
    values = {k.lower(): v for k, v in kwargs["headers"].items()}
    assert values["x-monocle-retrieve-traces"] == "caller-value"


def test_no_injection_when_env_unset(monkeypatch):
    monkeypatch.delenv("MONOCLE_TRACE_RETRIEVAL_KEY", raising=False)
    runner = HttpRunner()
    kwargs = {"method": "GET"}
    runner._maybe_inject_retrieval_key(kwargs)
    assert "headers" not in kwargs or all(
        k.lower() != "x-monocle-retrieve-traces" for k in kwargs.get("headers", {})
    )
```

Run: `pytest test_tools/tests/unit/test_http_runner_key_injection.py -v`
Expected: FAIL — `AttributeError: 'HttpRunner' object has no attribute '_maybe_inject_retrieval_key'`

- [ ] **Step 2: Implement the injection**

In `test_tools/src/monocle_test_tools/runner/http_runner.py`, add these imports at the top:

```python
import os
from monocle_apptrace.instrumentation.common.constants import (
    MONOCLE_TRACE_RETRIEVAL_KEY_ENV,
    TRACE_RETURN_REQUEST_HEADER,
)
```

Add the method to `class HttpRunner(_BaseHttpRunner):` and call it from `run_agent_async` before `super().run_agent_async`:

```python
    def _maybe_inject_retrieval_key(self, kwargs) -> None:
        """Add the trace-retrieval key header from MONOCLE_TRACE_RETRIEVAL_KEY,
        unless the caller already supplied the header."""
        key = os.environ.get(MONOCLE_TRACE_RETRIEVAL_KEY_ENV)
        if not key:
            return
        headers = dict(kwargs.get("headers") or {})
        if any(str(k).lower() == TRACE_RETURN_REQUEST_HEADER for k in headers):
            return  # caller-supplied header wins
        headers[TRACE_RETURN_REQUEST_HEADER] = key
        kwargs["headers"] = headers
```

Update `HttpRunner.run_agent_async` to inject before the request:

```python
    async def run_agent_async(self, root_agent: str, *args, **kwargs) -> Any:
        self._maybe_inject_retrieval_key(kwargs)
        response = await super().run_agent_async(root_agent, *args, **kwargs)
        self._capture_remote_spans(response)
        return response
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest test_tools/tests/unit/test_http_runner_key_injection.py -v`
Expected: PASS (3 tests)

- [ ] **Step 4: Run the existing HttpRunner test for no regression**

Run: `pytest test_tools/tests/unit/test_http_runner_ingest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add test_tools/src/monocle_test_tools/runner/http_runner.py test_tools/tests/unit/test_http_runner_key_injection.py
git commit -m "feat(test-tools): HttpRunner auto-injects trace-retrieval key from env"
```

---

### Task 5: Update the e2e integration test for the key-based flow

**Files:**
- Modify: `test_tools/tests/integration/test_http_trace_return.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the e2e proves the authorized path (key match) returns spans, and a wrong/absent key is denied — end-to-end.

Context: the current e2e sets `MONOCLE_ENABLE_TRACE_RETURN=true` at module import and sends `headers={"x-monocle-retrieve-traces": "true"}`. Under the new gate that value no longer authorizes. Update it to set the server default key and the client key, and rely on `HttpRunner` auto-injection (so the test no longer passes the header explicitly for the authorized case).

- [ ] **Step 1: Update the module-level env setup**

At the top of `test_http_trace_return.py`, where `os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "true"` is set (module import time), add the key envs:

```python
os.environ["MONOCLE_ENABLE_TRACE_RETURN"] = "true"
os.environ["MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY"] = "e2e-s3cret"   # server side
os.environ["MONOCLE_TRACE_RETRIEVAL_KEY"] = "e2e-s3cret"           # client side (HttpRunner injects)
```

(Keep them consistent with wherever the existing enable-var is set — if it's set inside the fixture rather than at module top, set these in the same place, before `MonocleValidator()` is constructed.)

- [ ] **Step 2: Update the authorized test to rely on key injection**

In `test_server_spans_returned_in_band`, remove the explicit `headers={"x-monocle-retrieve-traces": "true"}` from the `run_agent` call so the runner auto-injects the key:

```python
    response = validator.run_agent(
        server + "/chat", "http",
        method="POST", json={"q": "hi"},
    )
    assert response.json()["answer"] == "echo: hi"
    import json
    raw = getattr(response, "_monocle_remote_spans", None)
    assert raw is not None, "server did not piggyback spans on the HTTP response (no _monocle_remote_spans)"
    remote_names = [s.get("name") for s in json.loads(raw)]
    assert "answer_question" in remote_names, f"answer_question not in piggybacked spans: {remote_names}"
```

- [ ] **Step 3: Add a wrong-key denial test (discriminating)**

Add a test proving a wrong key is denied end-to-end:

```python
def test_wrong_key_denied(server):
    import requests
    r = requests.post(server + "/chat", json={"q": "x"},
                      headers={"x-monocle-retrieve-traces": "not-the-key"})
    assert r.json()["answer"] == "echo: x"          # body clean
    assert "x-monocle-traces" not in {k.lower() for k in r.headers.keys()}  # no trailer header
```

(Update the existing `test_inert_when_disabled` if it sent value "true"; the inert case is about `MONOCLE_ENABLE_TRACE_RETURN=false`, which is orthogonal to the key — leave its intent but ensure it still asserts no `x-monocle-traces` header.)

- [ ] **Step 4: Run the integration tests**

Run: `pytest test_tools/tests/integration/test_http_trace_return.py -v`
Expected: PASS (authorized returns spans; wrong-key denied; inert-when-disabled still passes).

- [ ] **Step 5: Prove the authorized test is discriminating**

Temporarily set the client key to a wrong value (e.g. `os.environ["MONOCLE_TRACE_RETRIEVAL_KEY"] = "wrong"`) at module top and re-run ONLY `test_server_spans_returned_in_band`; it must FAIL on the `_monocle_remote_spans is not None` assertion. Restore the correct key. Paste both outputs into the report.

- [ ] **Step 6: Commit**

```bash
git add test_tools/tests/integration/test_http_trace_return.py
git commit -m "test(test-tools): e2e trace retrieval via key-based authorization"
```

---

## Self-Review Notes

- **Spec coverage:** constants (Task 1); default callback + dispatch with all deny paths (Task 2); gate swap at both sites + removal of the old check (Task 3); HttpRunner key injection with caller-precedence (Task 4); e2e authorized + wrong-key denial, discriminating (Task 5).
- **Master switch preserved:** every gate site keeps `is_trace_return_enabled() and ...` — the env master switch is unchanged.
- **Client-only injection:** the key is added in `HttpRunner` only; the generic `requests` instrumentation is untouched (constraint honored).
- **No apptrace→test_tools import:** Task 4 lives in test_tools and imports apptrace constants (allowed direction).
