import logging
from threading import local
import json
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, get_exception_status_code, try_option, Option, MonocleSpanException
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler, HttpSpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from urllib.parse import unquote

logger = logging.getLogger(__name__)
MAX_DATA_LENGTH = 1000

def get_url(args) -> str:
    route_path: Option[str] = try_option(getattr, args[0], 'path')
    return route_path.unwrap_or("")

def get_method(args) -> str:
#    return args[0]['method'] if 'method' in args[0] else ""
    http_method: Option[str] = try_option(getattr, args[0], 'method')
    return http_method.unwrap_or("")

def get_params(args) -> dict:
    if len(args) > 0 and hasattr(args[0], "content") and hasattr(args[0].content, "_buffer"):
        data = args[0].content._buffer
        if len(data) > 0:
            message = json.loads(data[0].decode('utf-8'))
            # Return the input query/text for params
            query = message.get('text', '')
            return query
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

def extract_status(arguments) -> str:
    if arguments["exception"] is not None:
        return get_exception_status_code(arguments)
    result = arguments['result']
    if hasattr(result, 'status'):
        status = f"{result.status}"
        if status not in HTTP_SUCCESS_CODES:
            error_message = extract_response(result)
            raise MonocleSpanException(f"error: {status} - {error_message}", status)
    else:
        status = "success"
    return status

def aiohttp_pre_tracing(args):
    return extract_http_headers(args[0].headers)

def aiohttp_post_tracing(token):
    clear_http_scopes(token)

def aiohttp_skip_span(args) -> bool:
    if get_method(args) == "HEAD":
        return True
    return False

def get_route(args) -> str:
    try:
        return args[0].match_info.route.resource.canonical
    except Exception as e:
        return get_url(args)

def get_function_name(args) -> str:
    return args[0].match_info.handler.__name__

class aiohttpSpanHandler(HttpSpanHandler):

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return aiohttp_pre_tracing(args), None
    
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        aiohttp_post_tracing(token)

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        return aiohttp_skip_span(args)