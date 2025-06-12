# Monocle Trace API Usage

This document explains how to use the Monocle trace-based utilities: `monocle_trace`, `amonocle_trace`, `monocle_trace_method`, `start_trace`, and `stop_trace`.

## 1. `monocle_trace`
A context manager for starting a new trace (span) for all code executed within its block. Use in synchronous code.

**Example:**
```python
from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace

with monocle_trace(span_name="my_operation", attributes={"user.id": "user-123"}):
    result = my_function()
```

- `span_name` (optional): Name for the span. Defaults to "custom_span".
- `attributes` (optional): Dict of custom attributes to set on the span.
- `events` (optional): List of events to add to the span.

## 2. `amonocle_trace`
An async context manager for starting a new trace (span) for all code executed within its block. Use in async code.

**Example:**
```python
from monocle_apptrace.instrumentation.common.instrumentor import amonocle_trace

async def my_async_func():
    async with amonocle_trace(span_name="async_op"):
        await do_async_work()
```

## 3. `monocle_trace_method`
A decorator to automatically start and stop a trace for a function (sync or async). All spans created in the function will be part of the same trace.

**Example (sync):**
```python
from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_method

@monocle_trace_method(span_name="process_request")
def process_request(data):
    ...
```

**Example (async):**
```python
@monocle_trace_method(span_name="process_job")
async def process_job(data):
    ...
```

## 4. `start_trace` and `stop_trace`
Low-level API to manually start and stop a trace span. Useful for advanced scenarios or when you need to control the span lifecycle directly.

**Example:**
```python
from monocle_apptrace.instrumentation.common.instrumentor import start_trace, stop_trace

token = start_trace(span_name="manual_span", attributes={"foo": "bar"})
try:
    result = do_work()
    stop_trace(token, final_attributes={"result": result})
except Exception:
    stop_trace(token, final_attributes={"error": True})
    raise
```

- `start_trace` returns a token representing the context. Pass this token to `stop_trace` to end the span.
- You can set initial attributes/events on start, and final attributes/events on stop.

## Notes
- If `span_name` is omitted, the function name or a default is used.
- All spans created within the context or decorated function will share the same trace ID.
- You can add custom attributes and events to the root span using the context manager arguments or the start/stop API.
