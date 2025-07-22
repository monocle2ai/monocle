# Monocle Scope API Usage

This document explains how to use the Monocle scope-based tracing utilities: `monocle_trace_scope`, `amonocle_trace_scope`, and `monocle_trace_scope_method`.

## 1. `monocle_trace_scope`
A context manager for attaching a scope to all spans created within its block. Useful for synchronous code.

**Example:**
```python
from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_scope

with monocle_trace_scope("user_id", "user-123"):  # All spans in this block will have scope.user_id=user-123
    result = my_function()
```

## 2. `amonocle_trace_scope`
An async context manager for attaching a scope to all spans created within its block. Use in async code.

**Example:**
```python
from monocle_apptrace.instrumentation.common.instrumentor import amonocle_trace_scope

async def my_async_func():
    async with amonocle_trace_scope("session_id", "sess-456"):
        await do_async_work()
```

## 3. `monocle_trace_scope_method`
A decorator to automatically attach a scope to all spans created within a function (sync or async).

**Example (sync):**
```python
from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_scope_method

@monocle_trace_scope_method("request_id", "req-789")
def process_request(data):
    ...
```

**Example (async):**
```python
@monocle_trace_scope_method("job_id", "job-001")
async def process_job(data):
    ...
```

## Notes
- If `scope_value` is omitted, a random UUID will be generated and used as the value.
- You can nest scopes; all active scopes will be attached to spans.
- Scopes are propagated to all spans created within the context or decorated function.
