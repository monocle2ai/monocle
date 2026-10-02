"""httpx client instrumentation: trace context out, returned spans back.

The same job the `requests` metamodel does, for apps and SDKs that use httpx.
The details of a call live on the Request handed to `send`, not in kwargs.
"""
import logging

from opentelemetry.context import attach, detach, get_value, set_value
from opentelemetry.propagate import inject
from opentelemetry.trace.propagation import _SPAN_KEY

from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import _MONOCLE_SPAN_KEY, add_monocle_trace_state
from monocle_apptrace.instrumentation.metamodel.httpx import allowed_urls

logger = logging.getLogger(__name__)


def get_request(arguments) -> object:
    """The httpx Request being sent, whether it came positionally or by name."""
    args = arguments.get("args") or ()
    if args:
        return args[0]
    return (arguments.get("kwargs") or {}).get("request")


def get_method(arguments) -> str:
    return getattr(get_request(arguments), "method", "GET")


def get_route(arguments) -> str:
    url = getattr(get_request(arguments), "url", None)
    return f"{url.netloc.decode('utf-8')}{url.path}" if url is not None else ""


def get_params(arguments) -> str:
    url = getattr(get_request(arguments), "url", None)
    return url.query.decode("utf-8") if url is not None else ""


def get_body(arguments) -> dict:
    """Left out, as it is for `requests`. The caller's own spans hold what was sent."""
    return {}


def extract_status(result) -> str:
    return f"{getattr(result, 'status_code', '')}"


def extract_response(result) -> str:
    try:
        return result.text
    except Exception:          # a streamed response has no body read yet
        return ""


def httpx_skip_span(request, trace_all_urls: bool) -> bool:
    """Trace a call only if its host is in `MONOCLE_TRACE_PROPAGATATION_URLS`.

    Most httpx traffic in an app is its model provider and other SDKs, and
    tracing all of it would bury the calls that matter. A test can ask for all.
    """
    if trace_all_urls:
        return False
    url = str(getattr(request, "url", "") or "")
    for allowed_url in allowed_urls:
        if url.startswith(allowed_url.strip()):
            return False
    return True


class HttpxSpanHandler(SpanHandler):

    _trace_all_urls: bool = False

    @staticmethod
    def set_trace_all_urls_for_test(trace_all: bool):
        HttpxSpanHandler._trace_all_urls = trace_all

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        return httpx_skip_span(get_request({"args": args, "kwargs": kwargs}),
                               HttpxSpanHandler._trace_all_urls)

    def pre_task_processing(self, to_wrap, wrapped, instance, args, kwargs, span):
        """Put this trace on the request, so the far side continues it.

        Monocle keeps its current span under its own context key, so the span is
        presented under the OpenTelemetry key for `inject` to find.
        """
        request = get_request({"args": args, "kwargs": kwargs})
        headers = getattr(request, "headers", None)
        if headers is not None:
            token = None
            try:
                token = attach(set_value(_SPAN_KEY, get_value(_MONOCLE_SPAN_KEY)))
                carrier = {}
                add_monocle_trace_state(carrier)
                inject(carrier)
                for name, value in carrier.items():
                    headers[name] = value
            finally:
                if token is not None:
                    detach(token)
        super().pre_task_processing(to_wrap, wrapped, instance, args, kwargs, span)

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex,
                             span, parent_span):
        """Take any spans the server returned off the response body.

        A server with trace return on appends its spans after a delimiter. They
        are left on the response as `_monocle_remote_spans`, and reported to a
        collector for callers that never see the response. The body is restored.
        """
        try:
            headers = getattr(result, "headers", None)
            header_value = headers.get(TRACE_RETURN_RESPONSE_HEADER) if headers else None
            if header_value:
                delimiter = tr.parse_delimiter_from_header(header_value)
                body = getattr(result, "_content", None)
                if delimiter and isinstance(body, (bytes, bytearray)):
                    clean, payload = tr.split_body_and_trailer(bytes(body), delimiter)
                    if payload is not None:
                        result._content = clean
                        result._monocle_remote_spans = tr.decode_payload(payload)
                        tr.record_returned_spans(result._monocle_remote_spans)
        except Exception as e:
            logger.debug(f"trace-return strip failed: {e}")
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex,
                                     span, parent_span)
