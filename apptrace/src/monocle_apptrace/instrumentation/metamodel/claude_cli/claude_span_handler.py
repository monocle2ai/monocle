import datetime

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from opentelemetry.context import attach, detach, get_current, get_value, set_value, Context
from monocle_apptrace.instrumentation.common.constants import SPAN_START_TIME, SPAN_END_TIME, AGENT_SESSION
from monocle_apptrace.instrumentation.common.utils import set_scope

class ClaudeSpanHandler(SpanHandler):
    @staticmethod
    def _iso_to_ns(ts_str: str) -> int:
        """Convert an ISO 8601 timestamp string to nanoseconds since Unix epoch."""
        return int(datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() * 1e9)

    def _set_span_times(self, kwargs):
        # Set the span start and end times based on the arguments
        start_time = end_time = None
        token_context: Context = None
        if SPAN_START_TIME in kwargs:
            start_time = kwargs.pop(SPAN_START_TIME)
        if SPAN_END_TIME in kwargs:
            end_time = kwargs.pop(SPAN_END_TIME)
        if start_time:
            token_context = set_value(SPAN_START_TIME, ClaudeSpanHandler._iso_to_ns(start_time), token_context)
        if end_time:
            token_context = set_value(SPAN_END_TIME, ClaudeSpanHandler._iso_to_ns(end_time), token_context)
        if AGENT_SESSION in kwargs:
            token_context = set_value(AGENT_SESSION, kwargs.get(AGENT_SESSION), token_context)
            return set_scope(AGENT_SESSION, kwargs.get(AGENT_SESSION), token_context)
        else:
            # attach(None) corrupts the OTel context — fall back to current context
            return attach(token_context if token_context is not None else get_current())

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return self._set_span_times(kwargs), None