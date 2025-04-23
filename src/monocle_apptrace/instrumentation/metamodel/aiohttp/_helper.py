import logging
from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, try_option, Option
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
    route_path: Option[str] = try_option(getattr, args[0], 'path')
    return route_path.unwrap_or("")

def get_method(args) -> str:
#    return args[0]['method'] if 'method' in args[0] else ""
    http_method: Option[str] = try_option(getattr, args[0], 'method')
    return http_method.unwrap_or("")

def get_params(args) -> dict:
    params: Option[str] = try_option(getattr, args[0], 'query_string')
    return unquote(params.unwrap_or(""))

def get_body(args) -> dict:
    return ""

def extract_response(result) -> str:
    if hasattr(result, 'text'):
        response = result.text[0:max(result.text.__len__(), MAX_DATA_LENGTH)]
    else:   
        response = ""
    return response

def extract_status(instance) -> str:
    status = instance.status if hasattr(instance, 'status') else ""
    return status

def aiohttp_pre_tracing(args):
    token_data.current_token = extract_http_headers(args[0].headers)

def aiohttp_post_tracing():
    clear_http_scopes(token_data.current_token)
    token_data.current_token = None

class aiohttpSpanHandler(SpanHandler):

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        aiohttp_pre_tracing(args)
        return super().pre_tracing(to_wrap, wrapped, instance, args, kwargs)
    
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        aiohttp_post_tracing()
        return super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)
