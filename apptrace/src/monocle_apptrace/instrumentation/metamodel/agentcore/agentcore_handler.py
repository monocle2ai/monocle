import logging

from monocle_apptrace.instrumentation.common.constants import (
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

    Same idea as the HTTP trace-return path, adapted to how AgentCore builds a
    response. FastAPI appends the trailer to the response body as it is sent;
    AgentCore serializes whatever the entrypoint returned, so there is no such
    hook. The trailer is instead appended to the ``Response`` object that
    ``_handle_invocation`` produces, which happens after serialization and is
    mutable — the same shape the Lambda handler uses on its response dict.

    Unlike the HTTP path there is no per-request authorization: boto3 can only
    send parameters AgentCore models, so a caller cannot supply the
    ``x-monocle-retrieve-traces`` header. Returning traces is therefore opt-in
    on the deployment itself through ``MONOCLE_ENABLE_TRACE_RETURN``, which only
    whoever deploys the agent controls, and IAM already governs who may invoke
    it at all.
    """

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Tag this invocation's spans so the trace-return exporter keeps them.

        The scope has to be active while the agent runs, which is why this hooks
        ``_handle_invocation`` (the caller) rather than the entrypoint itself.
        """
        if not tr.is_trace_return_enabled():
            return None, None
        return set_scopes({TRACE_RETURN_SCOPE_NAME: "true"}), None

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
