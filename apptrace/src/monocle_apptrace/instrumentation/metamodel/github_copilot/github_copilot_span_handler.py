import datetime

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from opentelemetry.context import attach, get_current, set_value, Context
from monocle_apptrace.instrumentation.common.constants import (
    SPAN_START_TIME,
    SPAN_END_TIME,
    AGENT_SESSION,
)
from monocle_apptrace.instrumentation.common.utils import set_scope
from monocle_apptrace.instrumentation.common.agent_edit_context import apply_to_span


class GitHubCopilotSpanHandler(SpanHandler):
    @staticmethod
    def _iso_to_ns(ts_str: str) -> int:
        """Convert an ISO 8601 timestamp string to nanoseconds since Unix epoch."""
        return int(datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() * 1e9)

    def _set_span_times(self, kwargs):
        start_time = end_time = None
        token_context: Context = None
        if SPAN_START_TIME in kwargs:
            start_time = kwargs.pop(SPAN_START_TIME)
        if SPAN_END_TIME in kwargs:
            end_time = kwargs.pop(SPAN_END_TIME)
        if start_time:
            token_context = set_value(SPAN_START_TIME, GitHubCopilotSpanHandler._iso_to_ns(start_time), token_context)
        if end_time:
            token_context = set_value(SPAN_END_TIME, GitHubCopilotSpanHandler._iso_to_ns(end_time), token_context)
        if AGENT_SESSION in kwargs:
            token_context = set_value(AGENT_SESSION, kwargs.get(AGENT_SESSION), token_context)
            return set_scope(AGENT_SESSION, kwargs.get(AGENT_SESSION), token_context)
        return attach(token_context if token_context is not None else get_current())

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return self._set_span_times(kwargs), None

    def pre_task_processing(self, to_wrap, wrapped, instance, args, kwargs, span):
        super().pre_task_processing(to_wrap, wrapped, instance, args, kwargs, span)
        apply_to_span(span, kwargs)
