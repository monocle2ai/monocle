import logging
from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from urllib.parse import unquote
from opentelemetry.context import get_current
from opentelemetry.trace import Span, get_current_span
from opentelemetry.trace.propagation import _SPAN_KEY

logger = logging.getLogger(__name__)
MAX_DATA_LENGTH = 1000
token_data = local()
token_data.current_token = None

def get_route(args) -> str:
    return args[0]['PATH_INFO'] if 'PATH_INFO' in args[0] else ""

def get_method(args) -> str:
    return args[0]['REQUEST_METHOD'] if 'REQUEST_METHOD' in args[0] else ""

def get_params(args) -> dict:
    params = args[0]['QUERY_STRING'] if 'QUERY_STRING' in args[0] else ""
    return unquote(params)

def get_body(args) -> dict:
    body = {}
    return body

def extract_response(instance) -> str:
    if hasattr(instance, 'data') and hasattr(instance, 'content_length'):
        response = instance.data[0:max(instance.content_length, MAX_DATA_LENGTH)]
    else:   
        response = ""
    return response

def extract_status(instance) -> str:
    status = instance.status if hasattr(instance, 'status') else ""
    return status

def flask_pre_tracing(args):
    headers = dict()
    for key, value in args[0].items():
        if key.startswith("HTTP_"):
            new_key = key[5:].lower().replace("_", "-")
            headers[new_key] = value
    token_data.current_token = extract_http_headers(headers)

def flask_post_tracing():
    clear_http_scopes(token_data.current_token)
    token_data.current_token = None

class FlaskSpanHandler(SpanHandler):

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        flask_pre_tracing(args)
        return super().pre_tracing(to_wrap, wrapped, instance, args, kwargs)
    
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        flask_post_tracing()
        return super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)

class FlaskResponseSpanHandler(SpanHandler):
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        try:
            _parent_span_context = get_current()
            if _parent_span_context is not None:
                parent_span: Span = _parent_span_context.get(_SPAN_KEY, None)
                if parent_span is not None:
                    self.hydrate_events(to_wrap, wrapped, instance, args, kwargs, return_value, parent_span)
        except Exception as e:
            logger.info(f"Failed to propogate flask response: {e}")
        super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)
