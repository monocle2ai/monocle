import logging
from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, try_option, Option, MonocleSpanException
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from urllib.parse import unquote, urlparse, ParseResult


logger = logging.getLogger(__name__)
MAX_DATA_LENGTH = 1000

def get_url(kwargs) -> ParseResult:
    url_str = try_option(getattr, kwargs['req'], 'url')
    url = url_str.unwrap_or(None)
    if url is not None:
        return urlparse(url)
    else:
        return None

def get_route(kwargs) -> str:
    url:ParseResult = get_url(kwargs)
    if url is not None:
        return url.path

def get_method(kwargs) -> str:
#    return args[0]['method'] if 'method' in args[0] else ""
    http_method: Option[str] = try_option(getattr, kwargs['req'], 'method')
    return http_method.unwrap_or("")

def get_params(kwargs) -> dict:
    url:ParseResult = get_url(kwargs)
    if url is not None:
        return unquote(url.query)

def get_body(kwargs) -> dict:
    if hasattr(kwargs['req'], 'get_body'):
        response = kwargs.get_body()
        if isinstance(response, bytes):
            response = response.decode('utf-8', errors='ignore')
    else:
        response = ""
    return response

def extract_response(result) -> str:
    if hasattr(result, 'get_body'):
        response = result.get_body() #  text[0:max(result.text.__len__(), MAX_DATA_LENGTH)]
        if isinstance(response, bytes):
            response = response.decode('utf-8', errors='ignore')
    else:
        response = ""
    return response

def extract_status(result) -> str:
    status = f"{result.status_code}" if hasattr(result, 'status_code') else ""
    if status not in HTTP_SUCCESS_CODES:
        error_message = extract_response(result)
        raise MonocleSpanException(f"error: {status} - {error_message}")
    return status

def azure_func_pre_tracing(kwargs):
    headers = kwargs['req'].headers if hasattr(kwargs['req'], 'headers') else {}
    return extract_http_headers(headers)

def azure_func_post_tracing(token):
    clear_http_scopes(token)

class azureSpanHandler(SpanHandler):

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return azure_func_pre_tracing(kwargs)
    
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        azure_func_post_tracing(token)
