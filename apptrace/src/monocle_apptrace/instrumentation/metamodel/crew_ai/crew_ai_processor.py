import logging
from opentelemetry.context import set_value, attach, detach, get_value, get_current
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY, SCOPE_NAME, AGENT_SESSION
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.metamodel.crew_ai._helper import (
   DELEGATION_NAME_PREFIX, get_name, is_root_crew_name, is_delegation_task, CREW_AI_AGENT_NAME_KEY, is_streaming_mode
)
from monocle_apptrace.instrumentation.metamodel.crew_ai.entities.inference import (
    AGENT_DELEGATION, AGENT_REQUEST, AGENT
)
from monocle_apptrace.instrumentation.common.scope_wrapper import start_scope, stop_scope
from monocle_apptrace.instrumentation.common.utils import is_scope_set

logger = logging.getLogger(__name__)

class CrewAIAgentHandler(SpanHandler):
    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        """Skip the initial kickoff span when streaming is enabled.
        The actual execution will be traced through Task.execute_sync/execute_async"""
        if is_streaming_mode(instance) and to_wrap.get('method') in ['kickoff', 'kickoff_async']:
            logger.debug(f"Skipping span for streaming {to_wrap.get('method')} - tracing will happen at Task level")
            return True
        return False

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        # Skip context attachment in streaming mode — kickoff() returns lazily before tasks run.
        if is_streaming_mode(instance) and to_wrap.get('method') in ['kickoff', 'kickoff_async']:
            return None, None
        context = set_value(CREW_AI_AGENT_NAME_KEY, get_name(instance))
        context = set_value(AGENT_PREFIX_KEY, DELEGATION_NAME_PREFIX, context)
        scope_name = AGENT_REQUEST.get("type")
        if not is_scope_set(scope_name):
            agent_request_wrapper = to_wrap.copy()
            agent_request_wrapper["output_processor"] = AGENT_REQUEST
            # CrewAI has no thread/session id of its own; anchor the whole run under one
            # auto-generated agentic.session scope so its traces (incl. async-task threads)
            # share a session. Only at the outermost call, and never override a caller's.
            if not is_scope_set(AGENT_SESSION):
                return start_scope(AGENT_SESSION, context=context), agent_request_wrapper
            return attach(context), agent_request_wrapper
        else:
            return attach(context), None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        if token is not None:
            detach(token)

    # In multi agent scenarios, the root crew is the one that orchestrates the other agents. CrewAI generates an extra root level kickoff()
    # call on top of the supervisor agent kickoff().
    # This span handler resets the parent kickoff call as generic type to avoid duplicate attributes/events in supervisor span and this root span.

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)

# async_execution=True tasks run in a raw threading.Thread (spawned by Task.execute_async)
# that doesn't inherit the OTEL context (current span + session-scope baggage), so their
# spans would orphan into a session-less trace. Bridge it: capture get_current() in
# execute_async, re-attach in _execute_task_async. Keyed by the shared Task instance.
_ASYNC_TASK_CONTEXT = {}

class CrewAITaskHandler(SpanHandler):
    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        # Both are context-bridge only (no real work of their own) — emit no span.
        return to_wrap.get('method') in ('execute_async', '_execute_task_async')

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        method = to_wrap.get('method')
        if method == 'execute_async':
            # Calling thread: snapshot the live context for the worker thread to restore.
            try:
                _ASYNC_TASK_CONTEXT[id(instance)] = get_current()
            except Exception as e:
                logger.warning(f"Error capturing async task context: {e}")
            return None, None
        if method == '_execute_task_async':
            # Worker thread: restore the captured context so child spans nest under the turn.
            token = None
            ctx = _ASYNC_TASK_CONTEXT.pop(id(instance), None)
            if ctx is not None:
                token = attach(ctx)
            return token, None
        if is_delegation_task(instance):
            agent_request_wrapper = to_wrap.copy()
            agent_request_wrapper["output_processor"] = AGENT_DELEGATION
        else:
            agent_request_wrapper = None
        return None, agent_request_wrapper

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        # Detach the context token attached for _execute_task_async (None for others).
        if token is not None:
            detach(token)

    # CrewAI uses tasks to coordinate agent execution. The method is task execute() with different task types.
    # Hence we use a different output processor for task execute() to format the span as agentic.delegation when appropriate.
    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)

class CrewAIToolHandler(SpanHandler):
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        if token is not None:
            detach(token)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)
