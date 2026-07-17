from monocle_apptrace.instrumentation.common.constants import (
    AGENT_SESSION, AGENT_EXECUTION_ID, AGENT_REQUEST_SPAN_NAME,
)
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import (
    set_scope, set_scopes, remove_scope, is_scope_set, get_current_monocle_span,
)
from monocle_apptrace.instrumentation.metamodel.adk import _helper
from monocle_apptrace.instrumentation.metamodel.adk.entities.agent import AGENT_ORCHESTRATOR

class AdkSpanHandler(SpanHandler):
    """Custom span handler for ADK instrumentation that adds session_id and turn scopes."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Set session_id scope before tracing begins."""
        session_id_token = None

        # Check if this is a Runner instance to set session_id
        session_id = None
        if hasattr(instance, '__class__') and instance.__class__.__name__ == 'Runner':
            session_id = kwargs.get('session_id')

        agent_request_wrapper = None
        try:
            agent_class = instance.config_type.model_fields['agent_class'].default
            if agent_class in ['SequentialAgent', 'LoopAgent', 'ParallelAgent']:
                agent_request_wrapper = to_wrap.copy()
                agent_request_wrapper["output_processor"] = AGENT_ORCHESTRATOR

                # Set execution ID scope ONLY for ParallelAgent
                if agent_class == 'ParallelAgent':
                    # Set both session and execution scopes together if session is present
                    if session_id:
                        session_id_token = set_scopes({
                            AGENT_SESSION: session_id,
                            AGENT_EXECUTION_ID: None  # Auto-generate unique ID
                        })
                    else:
                        # Set only execution scope
                        session_id_token = set_scope(AGENT_EXECUTION_ID, None)
            elif session_id:
                # Set only session scope for non-parallel agents
                session_id_token = set_scope(AGENT_SESSION, session_id)
        except (ValueError, AttributeError):
            # If we still need to set session_id but agent type check failed
            if session_id and session_id_token is None:
                session_id_token = set_scope(AGENT_SESSION, session_id)

        # Bind the turn scope to ADK's invocation_id at the root agent, before child spans.
        turn_token = self._bind_turn_scope(instance, args)
        tokens = [t for t in (session_id_token, turn_token) if t is not None]
        return (tokens or None), agent_request_wrapper

    def _bind_turn_scope(self, instance, args):
        """Set the turn scope to ADK's invocation_id once, at the root agent call.

        invocation_id isn't known until inside run_async, so it can't be set up-front
        (the REQUEST entity skips the random builtin turn scope). args[0] is the ADK
        InvocationContext; sub-agents inherit the scope, so is_scope_set guards a re-set.
        Also stamps the REQUEST/turn span, which opened before the id existed.
        """
        if getattr(instance, '__class__', None) is not None and instance.__class__.__name__ == 'Runner':
            return None
        if is_scope_set(AGENT_REQUEST_SPAN_NAME) or not args:
            return None
        invocation_id = getattr(args[0], 'invocation_id', None)
        if not invocation_id:
            return None
        token = set_scope(AGENT_REQUEST_SPAN_NAME, invocation_id)
        request_span = get_current_monocle_span()
        if request_span is not None and request_span.is_recording():
            request_span.set_attribute(f"scope.{AGENT_REQUEST_SPAN_NAME}", invocation_id)
        return token

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token=None):
        # pre_tracing may return multiple scope tokens; detach in reverse (LIFO) order.
        if isinstance(token, list):
            for scope_token in reversed(token):
                remove_scope(scope_token)
        elif token:
            remove_scope(token)
