import logging
from threading import local
from unittest import result
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, get_exception_status_code, try_option, Option, \
    MonocleSpanException
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from urllib.parse import unquote, urlparse, ParseResult

logger = logging.getLogger(__name__)
MAX_DATA_LENGTH = 1000
token_data = local()
token_data.current_token = None

def get_url(args) -> str:
    event = args[1]
    host = event.get('headers', {}).get('Host', '')
    stage = event.get('requestContext', {}).get('stage', '')
    path = event.get('path', '')
    query_params = event.get('queryStringParameters', {})
    scheme = 'https' if event.get('headers', {}).get('X-Forwarded-Proto', 'http') == 'https' else 'http'
    url = f"{scheme}://{host}/{stage}{path}"
    if query_params:
        from urllib.parse import urlencode
        url += '?' + urlencode(query_params)
    return url

def get_route(args) -> str:
    event = args[1]
    route = event.get("path") or event.get("requestContext", {}).get("path")
    return route

def get_method(args) -> str:
    event = args[1]
    http_method = event.get("httpMethod") or event.get("requestContext", {}).get("httpMethod")
    return http_method


def get_params(args) -> dict:
    event = args[1]
    question = None
    query_params = event.get('queryStringParameters', {})
    if isinstance(query_params, dict):
        question = query_params.get('question')
    return question

def get_body(args) -> dict:
    event = args[1]
    body = event.get("body")
    return body

def extract_response(result) -> str:
    if isinstance(result, dict) and 'body' in result:
        response = result['body']
        if isinstance(response, bytes):
            response = response.decode('utf-8', errors='ignore')
    else:
        response = ""
    return response


def extract_status(arguments) -> str:
    if arguments["exception"] is not None:
        return get_exception_status_code(arguments)
    result = arguments['result']
    if isinstance(result, dict) and 'statusCode' in result:
        status = f"{result['statusCode']}"
        if status not in HTTP_SUCCESS_CODES:
            error_message = extract_response(result)
            raise MonocleSpanException(f"error: {status} - {error_message}", status)
    else:
        status = "success"
    return status

def lambda_func_pre_tracing(kwargs):
    headers = kwargs['event'].get('headers', {}) if 'event' in kwargs else {}
    return extract_http_headers(headers)


def lambda_func_post_tracing(token):
    clear_http_scopes(token)


class lambdaSpanHandler(SpanHandler):
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return lambda_func_pre_tracing(kwargs), None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value,token):
        lambda_func_post_tracing(token)
