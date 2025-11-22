from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import set_scope
from monocle_apptrace.instrumentation.metamodel.adk import _helper
from monocle_apptrace.instrumentation.metamodel.adk.entities.agent import AGENT_ORCHESTRATOR

class AdkSpanHandler(SpanHandler):
    """Custom span handler for ADK instrumentation that adds session_id scope."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Set session_id scope before tracing begins."""
        session_id_token = None

        if hasattr(instance, '__class__') and instance.__class__.__name__ == 'Runner':
            session_id = kwargs.get('session_id')
            if session_id:
                session_id_token = set_scope(AGENT_SESSION, session_id)

        agent_request_wrapper = None
        try:
            if instance.config_type.model_fields['agent_class'].default in ['SequentialAgent', 'LoopAgent', 'ParallelAgent']:
                agent_request_wrapper = to_wrap.copy()
                agent_request_wrapper["output_processor"] = AGENT_ORCHESTRATOR
        except (ValueError, AttributeError):
            pass

        return session_id_token, agent_request_wrapper