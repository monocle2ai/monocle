from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import set_scope, remove_scope
from monocle_apptrace.instrumentation.common.utils import get_current_monocle_span
from monocle_apptrace.instrumentation.metamodel.strands._helper import extract_session_id

__all__ = ["StrandsSpanHandler"]


class StrandsSpanHandler(SpanHandler):

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        """Avoid duplicate spans when Agent.__call__ internally invokes Agent.stream_async."""
        if to_wrap.get("package") == "strands.agent.agent" and to_wrap.get("method") == "stream_async":
            parent_span = get_current_monocle_span()
            parent_name = getattr(parent_span, "name", "") if parent_span else ""
            if parent_name.endswith("Agent.__call__"):
                return True
        return False

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        session_id_token = None

        if hasattr(instance, "__class__") and instance.__class__.__name__ == "Agent":
            session_id = extract_session_id(instance,kwargs)
            if session_id:
                session_id_token = set_scope(AGENT_SESSION, session_id)

        return session_id_token, None



