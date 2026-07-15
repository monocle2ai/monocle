from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION, CHILD_ERROR_CODE
from monocle_apptrace.instrumentation.common.span_handler import (
    NonFrameworkSpanHandler,
    SpanHandler,
    WORKFLOW_TYPE_MAP,
)
from monocle_apptrace.instrumentation.common.utils import remove_scope, set_scope
from monocle_apptrace.instrumentation.metamodel.openai._helper import ( extract_session_id_from_agents)

__all__ = ["OpenAISpanHandler", "OpenAIAgentsSpanHandler"]


class OpenAISpanHandler(NonFrameworkSpanHandler):
    """Span handler for core OpenAI SDK operations."""

    def is_teams_span_in_progress(self) -> bool:
        return self.is_framework_span_in_progress() and self.get_workflow_name_in_progress() == WORKFLOW_TYPE_MAP["teams.ai"]

    def is_llamaindex_span_in_progress(self) -> bool:
        return self.is_framework_span_in_progress() and self.get_workflow_name_in_progress() == WORKFLOW_TYPE_MAP["llama_index"]

    def skip_processor(self, to_wrap, wrapped, instance, span, args, kwargs) -> list[str]:
        if self.is_teams_span_in_progress():
            return ["attributes", "events.data.input", "events.data.output"]
        if self.is_llamaindex_span_in_progress():
            return []
        return super().skip_processor(to_wrap, wrapped, instance, span, args, kwargs)

    def hydrate_events(self, to_wrap, wrapped, instance, args, kwargs, ret_result, span, parent_span=None, ex: Exception = None, is_post_exec: bool = False) -> bool:
        if self.is_teams_span_in_progress() and ex is None:
            return super().hydrate_events(
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
                ret_result,
                span=parent_span,
                parent_span=None,
                ex=ex,
                is_post_exec=is_post_exec,
            )
        if self.is_llamaindex_span_in_progress():
            return super().hydrate_events(
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
                ret_result,
                span,
                parent_span=parent_span,
                ex=ex,
                is_post_exec=is_post_exec,
            )
        return super().hydrate_events(to_wrap, wrapped, instance, args, kwargs, ret_result, span, parent_span=parent_span, ex=ex, is_post_exec=is_post_exec)

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        if self.is_teams_span_in_progress() and ex is not None:
            if len(span.events) > 1 and span.events[1].name == "data.output" and span.events[1].attributes.get("error_code") is not None:
                parent_span.set_attribute(CHILD_ERROR_CODE, span.events[1].attributes.get("error_code"))
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)


class OpenAIAgentsSpanHandler(SpanHandler):
    """Custom span handler for OpenAI Agents Runner APIs that adds session scopes."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        session_id_token = None

        if hasattr(instance, "__class__") and instance.__class__.__name__ == "Runner":
            session_id = extract_session_id_from_agents(kwargs)
            if session_id:
                session_id_token = set_scope(AGENT_SESSION, session_id)

        return session_id_token, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        if token:
            remove_scope(token)

