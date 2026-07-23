# Design: Pluggable authorization callback for trace retrieval

**Date:** 2026-07-22
**Status:** Approved (brainstorming), pending implementation plan
**Builds on:** [2026-07-21-http-span-piggyback-design.md](2026-07-21-http-span-piggyback-design.md)

## Goal

Replace the naive per-request check that gates HTTP trace retrieval (currently
"the request header `x-monocle-retrieve-traces` equals the literal string
`true`") with a **pluggable authorization callback** the server operator
controls, plus a secure built-in default that compares the header value against
a server-side secret key. This hardens the feature so a server never returns its
internal spans to an unauthorized caller.

## Activation model

Two layers, unchanged master switch + new per-request authorization:

- **`MONOCLE_ENABLE_TRACE_RETURN=true`** remains the master switch. It gates
  exporter registration at `setup_monocle_telemetry` time and keeps the feature
  completely inert otherwise. Unchanged.
- The per-request check `is_trace_return_requested(headers)` (header ==
  `"true"`) is **replaced** by `is_trace_return_authorized(headers)`. Traces are
  returned for a request iff **enabled (env) AND authorized (callback)**.

The two per-request call sites are updated in place:
`extract_http_headers` (sets the `monocle_trace_return` scope) and the FastAPI
wiring in `fastapi/_helper.py` (installs the trailer-injection send wrapper).
`is_trace_return_enabled()` is unchanged.

## Server: authorization dispatch (`trace_return.py`)

### `is_trace_return_authorized(headers: dict) -> bool`

- If `MONOCLE_TRACE_RETRIEVAL_CALLBACK` is set to a non-empty value in the
  format `"pkg.module:callable"`:
  - Resolve it: split on the first `:`, `importlib.import_module(module_path)`,
    `getattr(module, attr)`, and verify the result is callable.
  - **On any failure** (bad format, `ModuleNotFoundError`, missing attribute,
    not callable): log a warning and **return `False`** — i.e. "if the module
    can't be loaded, don't enable trace retrieval."
  - Resolve per call. `importlib.import_module` is backed by `sys.modules`, so
    repeated resolution is cheap, and per-call resolution keeps behavior correct
    when tests toggle the env var. No custom caching.
- Otherwise (env unset/empty): use `default_trace_retrieval_callback`.
- Invoke the resolved callback as `callback(headers)`. **If it raises, log and
  return `False`.** Coerce the return to `bool`.

### `default_trace_retrieval_callback(headers: dict) -> bool`

- `key = os.environ.get(MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY)`. If unset or empty
  → return `False` (no key configured ⇒ deny).
- Look up `x-monocle-retrieve-traces` **case-insensitively** in `headers`. If
  absent → return `False`.
- Return `hmac.compare_digest(str(header_value), str(key))` (constant-time
  comparison to avoid timing side-channels).

### Removed

`is_trace_return_requested` is removed (superseded by
`is_trace_return_authorized`). The wire header stays `x-monocle-retrieve-traces`,
but its value semantics change from the literal `"true"` to **the retrieval
key**.

## Client: HttpRunner auto-injects the key

- New client-side env var `MONOCLE_TRACE_RETRIEVAL_KEY`.
- The **`HttpRunner`** (in `test_tools`) — and only `HttpRunner`, not the generic
  `requests` instrumentation — adds `x-monocle-retrieve-traces: <key>` to the
  outgoing request headers when `MONOCLE_TRACE_RETRIEVAL_KEY` is set **and** the
  caller has not already provided that header (caller-supplied value wins).
- Rationale for scoping to `HttpRunner`: the generic `requests` instrumentation
  fires for every outbound HTTP call in the app, so auto-injecting a secret
  there would broadcast the key to every host. `HttpRunner` is an explicit test
  driver aimed at a known target server, so injecting there is safe.

## New constants (`constants.py`)

| Constant | Value | Role |
|---|---|---|
| `MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV` | `"MONOCLE_TRACE_RETRIEVAL_CALLBACK"` | Server: `"pkg.mod:fn"` authorization callback |
| `MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY_ENV` | `"MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY"` | Server: secret key for the default callback |
| `MONOCLE_TRACE_RETRIEVAL_KEY_ENV` | `"MONOCLE_TRACE_RETRIEVAL_KEY"` | Client: key HttpRunner sends |

(`TRACE_RETURN_REQUEST_HEADER = "x-monocle-retrieve-traces"` already exists and
is reused.)

## Error handling summary

Every failure path denies safely, and a denied request sets no scope, so the
exporter never captures its spans and no trailer is emitted:

- Custom callback spec malformed / module not importable / attribute missing /
  not callable → `False` (+ warning).
- Custom callback raises at call time → `False` (+ warning).
- Default callback: key unset/empty, header absent, or value mismatch → `False`.

## Testing

**Unit — `default_trace_retrieval_callback`:**
- key set and header value matches → `True`.
- header value mismatches → `False`.
- header absent → `False`.
- `MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY` unset/empty → `False`.
- header key lookup is case-insensitive (`X-Monocle-Retrieve-Traces`).

**Unit — `is_trace_return_authorized` dispatch:**
- custom callback (a real `pkg.mod:fn` test target) is loaded and its bool is
  returned.
- unresolvable spec (bad module, missing attr, non-callable, malformed) → `False`.
- callback raises → `False`.
- no `MONOCLE_TRACE_RETRIEVAL_CALLBACK` → default callback is used.

**Unit — `HttpRunner` key injection:**
- with `MONOCLE_TRACE_RETRIEVAL_KEY` set, the request carries
  `x-monocle-retrieve-traces: <key>`.
- when the caller already set the header, the caller value is preserved.
- with the env unset, no header is added.

**Integration — update the existing e2e (`test_http_trace_return.py`):**
- server sets `MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY`; client sets
  `MONOCLE_TRACE_RETRIEVAL_KEY` (same value); rely on `HttpRunner`
  auto-injection.
- authorized request → spans returned (`response._monocle_remote_spans`
  populated, `answer_question` present).
- **discriminating negative:** a wrong or absent key → no `x-monocle-traces`
  header, no trailer, `_monocle_remote_spans` absent.

## Out of scope

- No change to the generic `requests` client instrumentation (it must not send
  the key).
- No change to the trailer wire format, exporter, or the FastAPI trailer
  injection mechanics — only the per-request gate function called by the two
  existing sites changes.
- Non-FastAPI server frameworks remain deferred to the piggyback Plan 2; when
  built, they will call the same `is_trace_return_authorized`.
