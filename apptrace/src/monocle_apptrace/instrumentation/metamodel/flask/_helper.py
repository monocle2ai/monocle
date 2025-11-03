import logging
from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, get_exception_status_code
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException
from urllib.parse import unquote
from opentelemetry.context import get_current
from opentelemetry.trace import Span, get_current_span
from opentelemetry.trace.propagation import _SPAN_KEY

logger = logging.getLogger(__name__)
MAX_DATA_LENGTH = 1000

def get_route(args) -> str:
    return args[0]['PATH_INFO'] if 'PATH_INFO' in args[0] else ""

def get_method(args) -> str:
    return args[0]['REQUEST_METHOD'] if 'REQUEST_METHOD' in args[0] else ""

def get_params(args) -> dict:
    params = args[0]['QUERY_STRING'] if 'QUERY_STRING' in args[0] else ""
    return unquote(params)

def get_url(args) -> str:
    url = ""
    if len(args) > 1 or not isinstance(args[0], dict):
        if 'HTTP_HOST' in args[0]:
            url = f"http://{args[0]['HTTP_HOST']}{args[0].get('REQUEST_URI', '')}"

    return url

def get_body(args) -> dict:
    return ""

def extract_response(instance) -> str:
    if hasattr(instance, 'data') and hasattr(instance, 'content_length'):
        response = instance.data[0:max(instance.content_length, MAX_DATA_LENGTH)]
    else:
        response = ""
    return response

def extract_status(arguments) -> str:
    if arguments["exception"] is not None:
        return get_exception_status_code(arguments)
    instance = arguments['instance']
    if hasattr(instance, 'status_code'):
        status = f"{instance.status_code}"
        if status not in HTTP_SUCCESS_CODES:
            error_message = extract_response(instance)
            raise MonocleSpanException(f"error: {status} - {error_message}", status)
    else:
        status = "success"
    return status

def flask_pre_tracing(args):
    headers = dict()
    for key, value in args[0].items():
        if key.startswith("HTTP_"):
            new_key = key[5:].lower().replace("_", "-")
            headers[new_key] = value
    return extract_http_headers(headers)

def flask_post_tracing(token):
    clear_http_scopes(token)

class FlaskSpanHandler(SpanHandler):

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return flask_pre_tracing(args)
    
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        flask_post_tracing(token)

class FlaskResponseSpanHandler(SpanHandler):
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        try:
            _parent_span_context = get_current()
            if _parent_span_context is not None:
                parent_span: Span = _parent_span_context.get(_SPAN_KEY, None)
                if parent_span is not None:
                    self.hydrate_events(to_wrap, wrapped, instance, args, kwargs, return_value, parent_span=parent_span)
        except Exception as e:
            logger.info(f"Failed to propogate flask response: {e}")
        super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value, token)
