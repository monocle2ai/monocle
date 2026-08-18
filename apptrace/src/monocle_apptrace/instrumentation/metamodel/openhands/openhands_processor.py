from opentelemetry import context as otel_context

from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import (
    get_current_monocle_span,
    get_scopes,
    set_scope,
)
from monocle_apptrace.instrumentation.metamodel.openhands._helper import extract_conversation_id

__all__ = ["OpenHandsSpanHandler", "OpenHandsToolHandler"]


class OpenHandsSpanHandler(SpanHandler):

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        # Anchor every run of a conversation to the same session using the
        # stable conversation id, so multi-turn runs share one session scope.
        # Delegated sub-conversations inherit the parent's session instead of
        # starting their own.
        session_id_token = None
        if get_scopes().get(AGENT_SESSION) is None:
            conversation_id = extract_conversation_id(instance)
            if conversation_id:
                session_id_token = set_scope(AGENT_SESSION, conversation_id)
        return session_id_token, None


def _get_action(args, kwargs):
    if args:
        return args[0]
    return kwargs.get("action")


def _action_key(action):
    # ActionEvent.id is the SDK's stable event UUID; fall back to object
    # identity for stand-ins without one.
    return getattr(action, "id", None) or id(action)


class OpenHandsToolHandler(SpanHandler):
    # Tool execution hops to worker threads (run_in_executor in the async path,
    # ThreadPoolExecutor in the sync parallel path), which do not propagate
    # contextvars. Contexts are stashed per action event at the caller-side
    # boundary and restored by the thread-side _run_safe, so tool spans and
    # anything nested in them (MCP calls, in-tool LLM calls, delegated
    # sub-conversations) parent correctly.
    _thread_hop_contexts = {}
    # Actions whose tool span is already open at the _arun_safe boundary; the
    # thread-side _run_safe call for the same action is a duplicate.
    _async_open_actions = set()

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        token = None
        method = to_wrap.get("method")
        if method == "execute_batch":
            # Sync path: stash before the batch fans out to threads. The tool
            # span itself is opened thread-side by _run_safe.
            for action in (args[0] if args else kwargs.get("action_events")) or []:
                self._thread_hop_contexts[_action_key(action)] = otel_context.get_current()
        elif method == "_run_safe":
            action = _get_action(args, kwargs)
            if action is not None:
                stashed = self._thread_hop_contexts.pop(_action_key(action), None)
                if stashed is not None and not get_current_monocle_span().get_span_context().is_valid:
                    token = otel_context.attach(stashed)
        return token, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token=None):
        if to_wrap.get("method") == "execute_batch":
            # Cancelled/skipped tools never reach _run_safe's pop.
            for action in (args[0] if args else kwargs.get("action_events")) or []:
                self._thread_hop_contexts.pop(_action_key(action), None)
        super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value, token=token)

    def pre_task_processing(self, to_wrap, wrapped, instance, args, kwargs, span):
        if to_wrap.get("method") == "_arun_safe":
            # Async path: stash with this tool span active so the thread-side
            # nested spans land under it.
            action = _get_action(args, kwargs)
            if action is not None:
                key = _action_key(action)
                self._thread_hop_contexts[key] = otel_context.get_current()
                self._async_open_actions.add(key)
        super().pre_task_processing(to_wrap, wrapped, instance, args, kwargs, span)

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        if to_wrap.get("method") == "_arun_safe":
            # Leak backstop for tools cancelled before _run_safe consumed the
            # stash; with stable keys an early pop can never re-parent a later
            # action into the wrong trace.
            action = _get_action(args, kwargs)
            if action is not None:
                key = _action_key(action)
                self._thread_hop_contexts.pop(key, None)
                self._async_open_actions.discard(key)
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        method = to_wrap.get("method")
        # execute_batch is a stash-only boundary, never a span.
        if method == "execute_batch":
            return True
        # Thread-side _run_safe for an action whose tool span _arun_safe
        # already opened: duplicate inner boundary.
        if method == "_run_safe":
            action = _get_action(args, kwargs)
            if action is not None and _action_key(action) in self._async_open_actions:
                return True
        # No context even after restore: dropping the span beats emitting an
        # orphan trace.
        return not get_current_monocle_span().get_span_context().is_valid
