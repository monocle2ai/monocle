import logging
from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, try_option, Option, \
    MonocleSpanException
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from urllib.parse import unquote, urlparse, ParseResult

logger = logging.getLogger(__name__)
MAX_DATA_LENGTH = 1000
token_data = local()
token_data.current_token = None

def get_url(kwargs) -> ParseResult:
    url_str = try_option(lambda k: k.get('path'), kwargs['event'])
    url = url_str.unwrap_or(None)
    if url is not None:
        return urlparse(url)
    else:
        return None

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


def extract_status(result) -> str:
    status = f"{result['statusCode']}" if isinstance(result, dict) and 'statusCode' in result else ""
    if status not in HTTP_SUCCESS_CODES:
        error_message = extract_response(result)
        raise MonocleSpanException(f"error: {status} - {error_message}")
    return status


def lambda_func_pre_tracing(kwargs):
    headers = kwargs['event'].get('headers', {}) if 'event' in kwargs else {}
    return extract_http_headers(headers)


def lambda_func_post_tracing(token):
    clear_http_scopes(token)


class lambdaSpanHandler(SpanHandler):
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return lambda_func_pre_tracing(kwargs)

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value,token):
        lambda_func_post_tracing(token)
