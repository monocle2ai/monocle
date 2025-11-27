import logging
from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, get_exception_status_code
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException
from opentelemetry.context import get_current
from opentelemetry.trace import Span
from opentelemetry.trace.propagation import _SPAN_KEY
import json
import urllib.parse

logger = logging.getLogger(__name__)
MAX_DATA_LENGTH = 1000
token_data = local()
token_data.current_token = None

def get_url(args) -> str:
    server = args.get('server', ('127.0.0.1', 80))
    host, port = server
    path = args.get('path', '/')
    scheme = args.get('scheme', 'http')
    return f"{scheme}://{host}:{port}{path}"

def get_route(scope) -> str:
    return scope.get('path', '')

def get_method(scope) -> str:
    return scope.get('method', '')

def get_params(args) -> dict:
    try:
        query_bytes = args.get("query_string", "")
        query_str = query_bytes.decode('utf-8')
        params = urllib.parse.parse_qs(query_str)
        question = params.get('question', [''])[0]
        return question
    except Exception as e:
        logger.warning(f"Error extracting params: {e}")
        return {}

def extract_response(response) -> str:
    try:
        if hasattr(response, 'body'):
            data = response.body
            answer = json.loads(data.decode("utf-8"))
            return answer
    except Exception as e:
        logger.warning(f"Error extracting response: {e}")
        return ""

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

def fastapi_pre_tracing(scope):
    headers = {k.decode('utf-8').lower(): v.decode('utf-8')
               for k, v in scope.get('headers', [])}
    token_data.current_token = extract_http_headers(headers)

def fastapi_post_tracing():
    clear_http_scopes(token_data.current_token)
    token_data.current_token = None

class FastAPISpanHandler(SpanHandler):
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        scope = args[0] if args else {}
        fastapi_pre_tracing(scope)
        return super().pre_tracing(to_wrap, wrapped, instance, args, kwargs)

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        fastapi_post_tracing()
        return super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value, token)

class FastAPIResponseSpanHandler(SpanHandler):
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        try:
            ctx = get_current()
            if ctx is not None:
                parent_span: Span = ctx.get(_SPAN_KEY)
                if parent_span is not None:
                    self.hydrate_events(to_wrap, wrapped, instance, args, kwargs,
                                        return_value, span=parent_span)
        except Exception as e:
            logger.info(f"Failed to propagate fastapi response: {e}")
        super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value, token)
