"""Unit regression tests for the CrewAI async-task fixes.

These cover two gaps surfaced by a multi-agent CrewAI demo that uses
`async_execution=True` tasks (which CrewAI runs in a raw threading.Thread):

1. `Task.execute_async` is a synchronous method returning a concurrent.futures.Future,
   NOT a coroutine. It must be wrapped with the sync `task_wrapper`; the async
   `atask_wrapper` turned the return value into a coroutine and CrewAI crashed on
   `future.result()` ("'coroutine' object has no attribute 'result'").

2. The work for async tasks runs in a spawned thread that does NOT inherit the OTEL
   context, so the agent/inference spans fragmented into orphan, session-less traces.
   `CrewAITaskHandler` bridges the context: it snapshots `get_current()` in
   `execute_async` and re-attaches it in `_execute_task_async` (the thread entry point),
   so the spans nest under the originating turn and inherit its session-scope baggage.

3. `extract_tool_response` must not stringify an object's `.output`/`.content` attribute
   when it is None (e.g. Exa's SearchResponse.output) — that produced the literal "None"
   and dropped the real payload. It now falls through to str(result).

None of these require a live LLM or network call.
"""
import threading

from opentelemetry import baggage
from opentelemetry.context import detach

from monocle_apptrace.instrumentation.common.utils import set_scope, MONOCLE_SCOPE_NAME_PREFIX
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper
from monocle_apptrace.instrumentation.metamodel.crew_ai.methods import CREW_AI_METHODS
from monocle_apptrace.instrumentation.metamodel.crew_ai.crew_ai_processor import (
    CrewAITaskHandler, _ASYNC_TASK_CONTEXT,
)
from monocle_apptrace.instrumentation.metamodel.crew_ai import _helper


def _by_method():
    return {m["method"]: m for m in CREW_AI_METHODS}


def test_execute_async_uses_sync_wrapper_not_coroutine():
    """Guards the crash regression: execute_async must use the sync wrapper."""
    methods = _by_method()
    assert "execute_async" in methods
    assert methods["execute_async"]["wrapper_method"] is task_wrapper
    assert methods["execute_async"]["wrapper_method"] is not atask_wrapper


def test_execute_task_async_thread_entry_is_registered():
    """The thread entry point must be wrapped so the context bridge runs."""
    methods = _by_method()
    assert "_execute_task_async" in methods
    assert methods["_execute_task_async"]["wrapper_method"] is task_wrapper
    assert methods["_execute_task_async"]["span_handler"] == "crew_ai_task_handler"


def test_async_methods_emit_no_span():
    """Both bridge methods are skip_span: they carry no real work themselves."""
    handler = CrewAITaskHandler()
    common = (None, object(), (), {})  # wrapped, instance, args, kwargs
    assert handler.skip_span({"method": "execute_async"}, *common) is True
    assert handler.skip_span({"method": "_execute_task_async"}, *common) is True
    # execute_sync still emits a span
    assert handler.skip_span({"method": "execute_sync"}, *common) is False


def test_context_bridge_propagates_scope_across_thread():
    """The session-scope baggage captured in execute_async is restored in the worker
    thread by _execute_task_async, instead of being lost (orphan, session-less trace)."""
    handler = CrewAITaskHandler()

    class FakeTask:
        pass

    instance = FakeTask()
    scope_key = f"{MONOCLE_SCOPE_NAME_PREFIX}meeting.session"

    # Main thread: establish a scope, then capture it via execute_async pre_tracing.
    token = set_scope("meeting.session", "sess-123")
    try:
        assert baggage.get_baggage(scope_key) == "sess-123"
        cap_token, alt = handler.pre_tracing({"method": "execute_async"}, None, instance, (), {})
        assert cap_token is None and alt is None
        assert id(instance) in _ASYNC_TASK_CONTEXT
    finally:
        detach(token)

    # Main-thread scope is now gone.
    assert baggage.get_baggage(scope_key) is None

    seen = {}

    def worker():
        # Fresh thread context: scope is NOT visible (this is the bug we fix).
        seen["before"] = baggage.get_baggage(scope_key)
        tkn, _alt = handler.pre_tracing({"method": "_execute_task_async"}, None, instance, (), {})
        seen["during"] = baggage.get_baggage(scope_key)
        handler.post_tracing({"method": "_execute_task_async"}, None, instance, (), {}, None, tkn)
        seen["after"] = baggage.get_baggage(scope_key)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert seen["before"] is None          # thread starts without the scope
    assert seen["during"] == "sess-123"    # context bridge restored it
    assert seen["after"] is None           # cleanly detached on post_tracing
    assert id(instance) not in _ASYNC_TASK_CONTEXT  # entry popped, no leak


def test_extract_tool_response_none_output_falls_through():
    """An object whose .output is None must not serialize to the literal 'None';
    fall through to str(result) so the real payload is kept (Exa SearchResponse case)."""
    class FakeSearchResponse:
        output = None          # exists but is None
        results = ["r1", "r2"]

        def __str__(self):
            return f"SearchResponse(results={self.results})"

    out = _helper.extract_tool_response(FakeSearchResponse())
    assert out != "None"
    assert "results=" in out


def test_extract_tool_response_uses_output_when_present():
    """When .output holds a value, it is still used (unchanged behavior)."""
    class WithOutput:
        output = "real-output"

    assert _helper.extract_tool_response(WithOutput()) == "real-output"


def test_extract_tool_response_plain_string():
    assert _helper.extract_tool_response("hello") == "hello"
    assert _helper.extract_tool_response(None) == ""
