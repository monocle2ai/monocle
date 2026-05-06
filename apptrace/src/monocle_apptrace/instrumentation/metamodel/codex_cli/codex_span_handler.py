import datetime

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from opentelemetry.context import attach, get_current, set_value, Context
from monocle_apptrace.instrumentation.common.constants import SPAN_START_TIME, SPAN_END_TIME, AGENT_SESSION, CODEX_TURN_SCOPE, CODEX_INVOCATION_SCOPE
from monocle_apptrace.instrumentation.common.utils import set_scopes


class CodexSpanHandler(SpanHandler):
    @staticmethod
    def _iso_to_ns(ts_str: str) -> int:
        return int(datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() * 1e9)

    def _prepare_span_context(self, kwargs):
        start_time = end_time = None
        token_context: Context = None
        if SPAN_START_TIME in kwargs:
            start_time = kwargs.pop(SPAN_START_TIME)
        if SPAN_END_TIME in kwargs:
            end_time = kwargs.pop(SPAN_END_TIME)
        if start_time:
            token_context = set_value(SPAN_START_TIME, CodexSpanHandler._iso_to_ns(start_time), token_context)
        if end_time:
            token_context = set_value(SPAN_END_TIME, CodexSpanHandler._iso_to_ns(end_time), token_context)
        scopes = {}
        if CODEX_TURN_SCOPE in kwargs:
            scopes[CODEX_TURN_SCOPE] = kwargs.pop(CODEX_TURN_SCOPE)
        if CODEX_INVOCATION_SCOPE in kwargs:
            scopes[CODEX_INVOCATION_SCOPE] = kwargs.pop(CODEX_INVOCATION_SCOPE)
        if AGENT_SESSION in kwargs:
            scopes[AGENT_SESSION] = kwargs.get(AGENT_SESSION)
        if scopes:
            return set_scopes(scopes, token_context)
        return attach(token_context if token_context is not None else get_current())

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return self._prepare_span_context(kwargs), None
