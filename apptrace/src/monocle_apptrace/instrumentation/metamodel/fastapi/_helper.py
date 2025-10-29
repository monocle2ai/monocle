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
    route = scope.get('path', '')
    if not route:
        route = scope.get('raw_path', b'').decode('utf-8')
    return route

def get_method(scope) -> str:
    return scope.get('method', scope.get('type', ''))

def get_params(args) -> dict:
    try:
        if isinstance(args, dict):
            query_bytes = args.get("query_string", "")
            if isinstance(query_bytes, bytes):
                query_str = query_bytes.decode('utf-8')
            else:
                query_str = str(query_bytes)
            params = urllib.parse.parse_qs(query_str)
            return params
        elif hasattr(args, 'query_params'):
            return dict(args.query_params)
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
    def __init__(self):
        super().__init__()
        self._trace_id = None
        self._root_span = None
        self._context_token = None

    async def on_pre_tracing(self, func, fn_args, fn_kwargs):
        """
        Pre tracing handler

        Args:
            func: Original function
            fn_args: Function args
            fn_kwargs: Function kwargs

        Returns:
            None
        """
        from opentelemetry import context, trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        from opentelemetry.context.context import Context
        
        # Get the request object from ASGI scope
        scope = fn_args[0] if fn_args else {}
        if not isinstance(scope, dict):
            return
            
        # Get current context before we start
        current_ctx = context.get_current()
        
        # Store current context for restoration
        self._parent_context = current_ctx
        
        # Create new context with scope token if we have one
        scope_token = getattr(scope, 'scope_token', None)
        if scope_token:
            new_ctx = Context(scope_token)
            self._context_token = context.attach(new_ctx)
            
        # Set span name
        self._span_name = self.force_request_span_name(fn_args)
        
        # Get active span to pass context
        span = trace.get_current_span()
        if span:
            # Ensure trace state is passed to child spans
            self._trace_id = format(span.get_span_context().trace_id, "032x")

        self._span_name = self.force_request_span_name(fn_args)

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        try:
            fastapi_post_tracing()
        finally:
            # Detach the context we attached in pre_tracing
            if hasattr(self, '_context_token'):
                from opentelemetry import context
                context.detach(self._context_token)
                del self._context_token
            # Clear stored context
            self._trace_id = None
            self._root_span = None
            self._parent_context = None
        return super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)

class FastAPIResponseSpanHandler(SpanHandler):
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        try:
            ctx = get_current()
            if ctx is not None:
                parent_span: Span = ctx.get(_SPAN_KEY)
                if parent_span is not None:
                    self.hydrate_events(to_wrap, wrapped, instance, args, kwargs,
                                        return_value, parent_span=parent_span)
        except Exception as e:
            logger.info(f"Failed to propagate fastapi response: {e}")
        super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)