import logging
from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, get_exception_status_code, try_option, Option, MonocleSpanException
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER
from urllib.parse import unquote, urlparse, ParseResult


logger = logging.getLogger(__name__)
MAX_DATA_LENGTH = 1000

def get_url(kwargs) -> ParseResult:
    url_str = try_option(getattr, kwargs['req'], 'url')
    url = url_str.unwrap_or(None)
    if url is not None:
        return url
    else:
        return None

def get_function_name(kwargs) -> str:
    context = kwargs.get('context', None)
    if context is not None and hasattr(context, 'function_name'):
        return context.function_name
    return ""
    

def get_route(kwargs) -> str:
    url_str = get_url(kwargs)
    if url_str is not None:
        url: ParseResult = urlparse(url_str)
        return url.path
    return ""

def get_method(kwargs) -> str:
#    return args[0]['method'] if 'method' in args[0] else ""
    http_method: Option[str] = try_option(getattr, kwargs['req'], 'method')
    return http_method.unwrap_or("")

def get_params(kwargs) -> dict:
    url_str = get_url(kwargs)
    if url_str is not None:
        url: ParseResult = urlparse(url_str)
        return unquote(url.query)
    return {}

def get_body(kwargs) -> dict:
    if hasattr(kwargs['req'], 'get_body'):
        response = kwargs['req'].get_body()
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

def extract_status(arguments) -> str:
    if arguments["exception"] is not None:
        return get_exception_status_code(arguments)
    result = arguments['result']
    if hasattr(result, 'status_code'):
        status = f"{result.status_code}"
        if status not in HTTP_SUCCESS_CODES:
            error_message = extract_response(result)
            raise MonocleSpanException(f"error: {status} - {error_message}", status)
    else:
        status = "Unknown"
    return status

def azure_func_pre_tracing(kwargs):
    headers = kwargs['req'].headers if hasattr(kwargs['req'], 'headers') else {}
    return extract_http_headers(headers)

def azure_func_post_tracing(token):
    clear_http_scopes(token)

class azureSpanHandler(SpanHandler):

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return azure_func_pre_tracing(kwargs), None
    
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        azure_func_post_tracing(token)

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        try:
            if hasattr(result, "get_body") and hasattr(result, "headers"):
                trace_id = span.get_span_context().trace_id if span is not None else 0
                payload = tr.get_response_trailer(trace_id)
                if payload is not None:
                    header_value, trailer = payload
                    body = result.get_body() or b""
                    if isinstance(body, str):
                        body = body.encode("utf-8")
                    new_body = body + trailer
                    # func.HttpResponse has no public body setter; mutate the
                    # name-mangled private buffer that get_body() reads.
                    setattr(result, "_HttpResponse__body", new_body)
                    result.headers[TRACE_RETURN_RESPONSE_HEADER] = header_value
        except Exception as e:
            logger.debug(f"azfunc trace-return injection skipped: {e}")
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)
