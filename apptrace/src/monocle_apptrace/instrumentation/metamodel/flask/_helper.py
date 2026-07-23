import io
import json
import logging
from urllib.parse import unquote
from opentelemetry.trace import Span
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, get_exception_status_code, with_tracer_wrapper
from monocle_apptrace.instrumentation.common.wrapper import monocle_wrapper
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler, HttpSpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException
from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import TRACE_RETURN_RESPONSE_HEADER, TRACE_RETURN_SCOPE_NAME
from monocle_apptrace.instrumentation.common.utils import is_scope_set, get_current_monocle_span

logger = logging.getLogger(__name__)

try:
    from monocle_apptrace.instrumentation.common.constants import DEFAULT_MAX_ATTRIBUTE_LENGTH
    MAX_DATA_LENGTH = DEFAULT_MAX_ATTRIBUTE_LENGTH
except ImportError:
    # Fallback if constant doesn't exist (backwards compatibility)
    MAX_DATA_LENGTH = 51200  # 50KB

def get_route(args) -> str:
    return args[0]['PATH_INFO'] if 'PATH_INFO' in args[0] else ""

def get_method(args) -> str:
    return args[0]['REQUEST_METHOD'] if 'REQUEST_METHOD' in args[0] else ""

def get_params(args) -> dict:
    params = args[0]['QUERY_STRING'] if 'QUERY_STRING' in args[0] else ""
    if params:
        return unquote(params)
    if 'werkzeug.request' in args[0] and hasattr(args[0]['werkzeug.request'],'query_string'):
        return unquote(args[0]['werkzeug.request'].query_string)


def get_url(args) -> str:
    url = ""
    if len(args) > 1 or not isinstance(args[0], dict):
        if 'HTTP_HOST' in args[0]:
            url = f"http://{args[0]['HTTP_HOST']}{args[0].get('REQUEST_URI', '')}"

    return url

def get_body(args) -> str:
    # The raw body is captured once in flask_task_wrapper and cached in
    # environ['_request_body'] (wsgi.input is a consume-once stream, so it must
    # not be read here or the Flask view loses the body).
    environ = args[0] if args else {}
    body = environ.get('_request_body', b'') if isinstance(environ, dict) else b''
    if not body:
        return ""
    try:
        text_body = body.decode('utf-8')
    except Exception as e:
        logger.warning(f"Error decoding request body: {e}")
        return ""
    try:
        data = json.loads(text_body)
        if isinstance(data, (dict, list)):
            return json.dumps(data)[:MAX_DATA_LENGTH] if data else ""
        return str(data)[:MAX_DATA_LENGTH]
    except (json.JSONDecodeError, TypeError):
        return unquote(text_body)[:MAX_DATA_LENGTH]

@with_tracer_wrapper
def flask_task_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs):
    """Wraps Flask.wsgi_app to capture the request body into environ['_request_body']
    BEFORE span hydration runs. wsgi.input is a consume-once stream, so reading it for
    instrumentation would steal the body from the Flask view (causing 400s). We read it
    eagerly, cache the bytes, and replace wsgi.input with a fresh BytesIO so the app can
    still read it."""
    environ = args[0] if args else None
    if isinstance(environ, dict) and 'wsgi.input' in environ and '_request_body' not in environ:
        try:
            content_length = int(environ.get('CONTENT_LENGTH') or 0)
        except (ValueError, TypeError):
            content_length = 0
        if content_length > 0:
            try:
                body = environ['wsgi.input'].read(content_length)
                environ['_request_body'] = body
                # rewind: hand the downstream app a fresh stream with the same bytes
                environ['wsgi.input'] = io.BytesIO(body)
            except Exception as e:
                logger.warning(f"Error pre-reading request body: {e}")
    return monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)

def _flask_wrap_start_response(start_response, header_value, extra_len):
    """Add the trace-return header and bump Content-Length (buffered) on the WSGI start_response.
    Wraps the WSGI-level callable rather than mutating the werkzeug Response object, so the
    response span's captured data.output (read from the untouched Response) stays clean."""
    def wrapped(status, headers, exc_info=None):
        new_headers = []
        for name, value in headers:
            if extra_len and name.lower() == "content-length":
                value = str(int(value) + extra_len)
            new_headers.append((name, value))
        new_headers.append((TRACE_RETURN_RESPONSE_HEADER, header_value))
        return start_response(status, new_headers, exc_info)
    return wrapped


def _flask_chain_trailer(app_iter, trailer):
    for chunk in app_iter:
        yield chunk
    if trailer:
        yield trailer


def _flask_wrap_stream(app_iter, trace_id, delimiter):
    for chunk in app_iter:
        yield chunk
    trailer = tr.pop_and_build_trailer(trace_id, delimiter)
    if trailer:
        yield trailer


@with_tracer_wrapper
def flask_response_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs):
    """Wrap werkzeug Response.__call__ to append the trace-return trailer at the WSGI boundary
    (start_response headers + app_iter), never mutating the werkzeug Response object itself, so
    the response span's captured data.output stays the clean, un-trailered body.
    Buffered: bump Content-Length via start_response, chain the trailer onto app_iter.
    Streamed: set the header up front (no Content-Length to fix), wrap app_iter to append a
    final trailer chunk."""
    eligible = tr.is_trace_return_enabled() and is_scope_set(TRACE_RETURN_SCOPE_NAME)
    if not eligible or len(args) < 2:
        return monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)
    environ, start_response = args[0], args[1]
    trace_id = get_current_monocle_span().get_span_context().trace_id
    if not getattr(instance, "is_streamed", False):
        payload = tr.get_response_trailer(trace_id)
        if payload is None:
            return monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)
        header_value, trailer = payload
        new_sr = _flask_wrap_start_response(start_response, header_value, len(trailer))
        new_args = (environ, new_sr) + tuple(args[2:])
        app_iter = monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, new_args, kwargs)
        return _flask_chain_trailer(app_iter, trailer)
    # streamed: no Content-Length; header up front, trailer chunk at end
    delimiter = tr.make_delimiter()
    header_value = tr.build_response_header_value(delimiter)
    new_sr = _flask_wrap_start_response(start_response, header_value, 0)
    new_args = (environ, new_sr) + tuple(args[2:])
    app_iter = monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, new_args, kwargs)
    return _flask_wrap_stream(app_iter, trace_id, delimiter)


def extract_response(instance) -> str:
    if hasattr(instance, 'data') and hasattr(instance, 'content_length'):
        response = instance.data[0:max(instance.content_length, MAX_DATA_LENGTH)]
        if isinstance(response, bytes):
            response = response.decode('utf-8')
        try:
            parsed = json.loads(response)
            if isinstance(parsed, (dict, list)) and not parsed:
                return ""
        except (json.JSONDecodeError, TypeError):
            pass
        return unquote(response) if response else ""
    else:
        response = ""
    return response

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
        status = "Unknown"
    return status

def flask_pre_tracing(args):
    headers = dict()
    for key, value in args[0].items():
        if key.startswith("HTTP_"):
            new_key = key[5:].lower().replace("_", "-")
            headers[new_key] = value
    return extract_http_headers(headers)

def flask_post_tracing(token):
    clear_http_scopes(token)

class FlaskSpanHandler(HttpSpanHandler):

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return flask_pre_tracing(args), None
    
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        flask_post_tracing(token)

class FlaskResponseSpanHandler(SpanHandler):
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
            logger.info(f"Failed to propagate flask response: {e}")
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)
