# Design: Extend trace-return injection to all HTTP server frameworks

**Date:** 2026-07-23
**Status:** Approved (brainstorming), pending implementation plan
**Builds on:** [2026-07-21-http-span-piggyback-design.md](2026-07-21-http-span-piggyback-design.md),
[2026-07-22-trace-retrieval-authorization-callback-design.md](2026-07-22-trace-retrieval-authorization-callback-design.md)

## Goal

The HTTP span-piggyback feature (return a server's Monocle child spans in-band on
the HTTP response) is implemented for **FastAPI** only. Extend the server-side
**response injection** to the remaining instrumented HTTP server frameworks:
**Flask, aiohttp, AWS Lambda, and Azure Functions**.

## What already works (no change)

The authorization + span-capture half is already framework-agnostic:

- Every framework's `pre_tracing` calls `extract_http_headers(headers)`, which
  runs `is_trace_return_authorized(headers)` and sets the
  `monocle_trace_return` scope when authorized.
- Child spans of an authorized request are therefore already captured into the
  process-global `TraceReturnSpanExporter` for **all** frameworks.
- The client side (`HttpRunner` key injection + `RequestSpanHandler` trailer
  strip) is transport-agnostic and already works for any HTTP response.

The only missing piece per framework is appending the trailer (and the
`x-monocle-traces` header) to the outgoing response. Adding it also fixes a
latent issue: with the feature enabled, these frameworks currently
capture-but-never-evict a trace's spans (no injection ever pops them), so the
exporter buffer would grow. Injection pops+evicts per response.

## Shared core (new, in `trace_return.py`)

Trailer-building is currently a method on `HttpSpanHandler`
(`build_trace_return_trailer`). Lambda and Azure handlers subclass plain
`SpanHandler`, so move the logic to framework- and class-agnostic module
functions:

- `pop_and_build_trailer(trace_id: int, delimiter: str) -> bytes | None` — pop
  this trace's captured spans from the exporter and build the trailer bytes;
  `None` when there is nothing to return. (This is the current body of
  `HttpSpanHandler.build_trace_return_trailer`, which now delegates to it — so
  FastAPI is unchanged.)
- `get_response_trailer(trace_id: int) -> tuple[str, bytes] | None` —
  convenience for **buffered** injection. Returns
  `(header_value, trailer_bytes)` where `header_value =
  build_response_header_value(make_delimiter())`, or `None` when disabled / no
  spans. One call yields both the header value and the body bytes.

Streaming paths keep using the primitives separately (`make_delimiter` +
`build_response_header_value` up front to set the header before the body starts,
then `pop_and_build_trailer` at end-of-stream) — the same shape FastAPI already
uses, because the header must be sent before the body but the spans are only
complete at the end.

The trace_id comes from the request span that the wrapper passes to
`post_task_processing` (`span.get_span_context().trace_id`) for the
object-mutation paths, or from `get_current_monocle_span()` where injection runs
outside `post_task_processing` (Flask/aiohttp streaming). Using
`get_current_monocle_span()` (not raw `get_current_span()`) matters because
Monocle isolates its span hierarchy under `MONOCLE_ISOLATE_SPANS=true`.

## Per-framework injection

### AWS Lambda (buffered only)

`lambdaSpanHandler` currently has no `post_task_processing`. Add one. `result` is
the API Gateway-style dict `{statusCode, headers, body}`. When
`get_response_trailer(trace_id)` returns a payload: append `trailer` (decoded to
str) to `result["body"]` and set `result["headers"]["x-monocle-traces"]`
(creating the `headers` dict if absent). Guard for non-string / base64 bodies by
skipping injection with a debug log. No streaming.

### Azure Functions (buffered only)

`azureSpanHandler.post_task_processing`: `result` is a `func.HttpResponse` whose
body is set at construction and has no public setter. Rebuild it:
`func.HttpResponse(body=old_body + trailer, status_code=result.status_code,
headers={**result.headers, "x-monocle-traces": header_value},
mimetype=result.mimetype)` and substitute it as the handler's return value.

Plan will confirm the wrapper supports substituting the result from
`post_task_processing`; if it does not, mutate the private body buffer as a
fallback. No streaming.

### Flask (buffered + streaming)

Inject in `FlaskResponseSpanHandler` (wraps werkzeug `Response.__call__`), using
`get_current_monocle_span()` for the trace_id:

- **buffered** (`not response.is_streamed`): before `__call__` produces output,
  `response.set_data(response.get_data() + trailer)` — werkzeug recomputes
  `Content-Length` — and set `response.headers["x-monocle-traces"]`.
- **streamed** (`response.is_streamed`): set the header (with the pre-generated
  delimiter) on `response.headers` before output starts, and wrap the WSGI
  `app_iter` returned by `__call__` to yield one final trailer chunk after the
  real body (built via `pop_and_build_trailer` at end-of-iteration). No
  `Content-Length` on streamed responses, so appending is safe.

### aiohttp (buffered + streaming)

`aiohttpSpanHandler` wraps `Application._handle`, which returns the response
before the writer serializes it.

- **buffered** `web.Response`: in `post_task_processing`, append to
  `result.body` and set `result.headers["x-monocle-traces"]`.
- **streamed** `web.StreamResponse`: the body is written during the handler, so
  post-mutation is too late. Instrument `StreamResponse.prepare` (add the
  `x-monocle-traces` header before headers are sent) and `StreamResponse.write_eof`
  (write the trailer bytes just before EOF). This is the fiddliest hook.

Plan will verify the `StreamResponse.prepare`/`write_eof` hooks; if no clean hook
exists, degrade aiohttp to **buffered-only** with a documented limitation rather
than ship a fragile streaming path.

## FastAPI

Behavior unchanged. Its ASGI send-wrapper is refactored only to call the shared
`pop_and_build_trailer` instead of `HttpSpanHandler.build_trace_return_trailer`
directly (no functional change; keeps one implementation of the pop+build).

## Testing

**Unit (per framework):**
- With a captured span in the exporter for a trace_id, the framework's injection
  produces a response carrying the `x-monocle-traces` header and the trailer
  appended after the body (buffered).
- Flask/aiohttp streaming: the wrapped `app_iter` / `write_eof` path appends
  exactly one trailer chunk after the real body, and the header is present.
- Deny/inert: when disabled or unauthorized (no captured spans), no header and
  the response body is unchanged.

**Integration:**
- Flask (WSGI test server) and aiohttp (test server) driven by `HttpRunner`,
  asserting `response._monocle_remote_spans` contains the server-side child span
  (discriminating — proven to fail if the feature is disabled), same pattern as
  the FastAPI e2e.
- Lambda / Azure Functions: invoke the route wrapper with a fake event / request
  object, assert the returned dict / `HttpResponse` carries the trailer and
  header, then that a client-side strip (`split_body_and_trailer` +
  `decode_payload`) recovers the span JSON.

**Regression:** existing per-framework instrumentation tests still pass; FastAPI
trace-return tests unchanged.

## Out of scope

- No change to the authorization model, exporter, codec, or client side.
- Lambda and Azure Functions do not stream (dict / object returns), so no
  streaming there.
- Non-HTTP frameworks are unaffected.

## Naming / wire format

Unchanged from the existing feature: response header `x-monocle-traces`
(`v1; delim=<uuid>`), scope `monocle_trace_return`, env vars
`MONOCLE_ENABLE_TRACE_RETURN` / `MONOCLE_TRACE_RETRIEVAL_*`.
