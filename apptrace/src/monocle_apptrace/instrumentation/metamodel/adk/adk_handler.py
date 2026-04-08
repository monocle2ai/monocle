from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION, AGENT_EXECUTION_ID
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import set_scope, set_scopes
from monocle_apptrace.instrumentation.metamodel.adk import _helper
from monocle_apptrace.instrumentation.metamodel.adk.entities.agent import AGENT_ORCHESTRATOR

class AdkSpanHandler(SpanHandler):
    """Custom span handler for ADK instrumentation that adds session_id scope."""

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

        return session_id_token, agent_request_wrapper