import logging
from collections import deque

from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, get_exception_status_code, with_tracer_wrapper
from monocle_apptrace.instrumentation.common.wrapper import amonocle_wrapper
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler, HttpSpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException
from opentelemetry.trace import Span
import json
import urllib.parse

logger = logging.getLogger(__name__)
MAX_DATA_LENGTH = 1000


async def _buffer_request_body(scope, receive):
    messages = []
    body_chunks = []

    while True:
        message = await receive()
        messages.append(message)
        if message.get("type") != "http.request":
            break

        body_chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break

    scope["_request_body"] = b"".join(body_chunks)
    replay_messages = deque(messages)

    async def _receive():
        if replay_messages:
            return replay_messages.popleft()
        return await receive()

    return _receive

@with_tracer_wrapper
async def fastapi_atask_wrapper(tracer, handler, to_wrap, wrapped, instance,
                               source_path, args, kwargs):
    """Wraps APIRoute.handle to capture POST body into scope['_request_body'],
    similar to how Werkzeug stores request data in environ['werkzeug.request']."""
    scope, receive = args[0], args[1]
    if scope.get('method', 'GET') in ('POST', 'PUT', 'PATCH'):
        args = (scope, await _buffer_request_body(scope, receive)) + args[2:]
    return await amonocle_wrapper(
        tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs
    )

def get_url(args) -> str:
    server = args.get('server', ('127.0.0.1', 80))
    host, port = server
    path = args.get('path', '/')
    scheme = args.get('scheme', 'http')
    return f"{scheme}://{host}:{port}{path}"

def get_route(scope) -> str:
    route = scope.get('route')
    if route is not None and getattr(route, 'path', None):
        return route.path
    return scope.get('path', '')

def get_method(scope) -> str:
    return scope.get('method', '')

def get_params(args) -> dict:
    try:
        query_bytes = args.get("query_string", b"")
        query_str = query_bytes.decode('utf-8') if isinstance(query_bytes, bytes) else str(query_bytes)
        if query_str:
            return urllib.parse.unquote(query_str)
        return ""

    except Exception as e:
        logger.warning(f"Error extracting params: {e}")
        return {}

def get_body(args) -> str:
    try:
        body = args.get('_request_body', b'')
        if body:
            return (body.decode('utf-8') if isinstance(body, bytes) else str(body))[:MAX_DATA_LENGTH]
        return ""
    except Exception as e:
        logger.warning(f"Error extracting body: {e}")
        return ""

def extract_response(response) -> str:
    try:
        if hasattr(response, 'body'):
            data = response.body
            answer = json.loads(data.decode("utf-8"))
            if isinstance(answer, (dict, list)):
                if not answer:
                    return ""
                return json.dumps(answer)[:MAX_DATA_LENGTH]
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
    return extract_http_headers(headers)

def fastapi_post_tracing(token):
    clear_http_scopes(token)

class FastAPISpanHandler(HttpSpanHandler):
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        scope = args[0] if args else {}
        return fastapi_pre_tracing(scope), None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        fastapi_post_tracing(token)

class FastAPIResponseSpanHandler(SpanHandler):
    # This span is only used to collect the data.input and data.output events and merge with parent span.
    # It's never exported by itself.
    def should_sample(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span:Span, parent_span:Span) -> bool:
        return False

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span:Span, parent_span:Span):
        try:
            if parent_span is not None:
                self.hydrate_events(to_wrap, wrapped, instance, args, kwargs,
                                    result, span=parent_span, is_post_exec=False)
                self.hydrate_events(to_wrap, wrapped, instance, args, kwargs,
                                    result, span=parent_span, is_post_exec=True)
        except Exception as e:
            logger.info(f"Failed to propagate fastapi response: {e}")
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)
