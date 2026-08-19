import logging

from monocle_apptrace.instrumentation.common.constants import (
    AGENTCORE_CUSTOM_HEADER_PREFIX,
    TRACE_RETURN_RESPONSE_HEADER,
    TRACE_RETURN_SCOPE_NAME,
)
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import (
    get_current_monocle_span,
    remove_scopes,
    set_scopes,
)

logger = logging.getLogger(__name__)

__all__ = ["AgentCoreSpanHandler"]


class AgentCoreSpanHandler(SpanHandler):
    """Returns an invocation's spans to the caller inside the AgentCore response.

    AgentCore serializes whatever the entrypoint returned, so the trailer is
    appended to the ``Response`` object ``_handle_invocation`` builds — after
    serialization, and mutable.

    Spans are returned only when the deployment opted in with
    ``MONOCLE_ENABLE_TRACE_RETURN``, declared the key it accepts in
    ``MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY``, and the caller presented that key.
    The key arrives under AgentCore's
    ``X-Amzn-Bedrock-AgentCore-Runtime-Custom-`` prefix, since those are the only
    caller headers AgentCore forwards to the agent; the prefix is stripped before
    ``is_trace_return_authorized`` decides.
    """

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Tag this invocation's spans so the trace-return exporter keeps them.

        The scope has to be active while the agent runs, which is why this hooks
        ``_handle_invocation`` (the caller) rather than the entrypoint itself.

        An unauthorized caller is refused here rather than at injection time.
        Without the scope the exporter captures nothing for this trace, so there
        is no trailer to build later — and the decision stays with the request
        that made it, instead of on a handler shared across concurrent
        invocations.
        """
        if not tr.is_trace_return_enabled():
            return None, None
        if not tr.is_trace_return_authorized(self._request_headers(args, kwargs)):
            return None, None
        return set_scopes({TRACE_RETURN_SCOPE_NAME: "true"}), None

    @staticmethod
    def _request_headers(args, kwargs) -> dict:
        """Headers of the invocation, with AgentCore's custom prefix removed.

        ``_handle_invocation(request)`` receives the Starlette request, so the
        headers as the caller sent them are available. Prefixed names are also
        kept under their original form, so a callback configured with
        ``MONOCLE_TRACE_RETRIEVAL_CALLBACK`` can still see what actually arrived.
        Returns an empty dict for anything unrecognizable, which denies.
        """
        request = kwargs.get("request") or (args[0] if args else None)
        raw = getattr(request, "headers", None)
        if raw is None:
            return {}
        try:
            items = list(raw.items())
        except Exception:
            return {}

        headers = dict(items)
        # Stripped names are added in a second pass so a forwarded header always
        # wins over one the caller happened to send under the same bare name,
        # rather than whichever of the two arrived last.
        prefix = AGENTCORE_CUSTOM_HEADER_PREFIX.lower()
        for name, value in items:
            if str(name).lower().startswith(prefix):
                headers[str(name)[len(prefix):]] = value
        return headers

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token=None):
        if token is not None:
            remove_scopes(token)

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        try:
            self._inject_trailer(result, span)
        except Exception as e:
            # Returning traces is a diagnostic convenience; never let it affect
            # the response the agent actually produced.
            logger.debug(f"agentcore trace-return injection skipped: {e}")
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)

    @staticmethod
    def _inject_trailer(result, span) -> None:
        """Append the trailer to a buffered Response, leaving its body otherwise intact.

        Skipped for streaming responses, which have no ``body`` to extend, and
        when this trace produced no captured spans.
        """
        if not tr.is_trace_return_enabled():
            return
        body = getattr(result, "body", None)
        if not isinstance(body, (bytes, bytearray)):
            return

        current_span = span if span is not None else get_current_monocle_span()
        trace_id = current_span.get_span_context().trace_id if current_span is not None else 0
        payload = tr.get_response_trailer(trace_id)
        if payload is None:
            return
        header_value, trailer = payload

        result.body = bytes(body) + trailer
        # Starlette fixed Content-Length when the Response was built, so it has
        # to be corrected or the trailer is truncated off the wire.
        result.headers["content-length"] = str(len(result.body))
        result.headers[TRACE_RETURN_RESPONSE_HEADER] = header_value
