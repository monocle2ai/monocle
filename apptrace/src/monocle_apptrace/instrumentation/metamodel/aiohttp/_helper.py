import logging
from threading import local
import json
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, get_exception_status_code, try_option, Option, MonocleSpanException, with_tracer_wrapper, is_scope_set, get_current_monocle_span
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler, HttpSpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES, TRACE_RETURN_RESPONSE_HEADER, TRACE_RETURN_SCOPE_NAME
from monocle_apptrace.instrumentation.common import trace_return as tr
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
            if query:
                return query
            return message.get('question','')
    params: Option[str] = try_option(getattr, args[0], 'query_string')
    return unquote(params.unwrap_or(""))

def get_body(args) -> str:
    if len(args) == 0:
        return ""
    request = args[0]
    # Use proper aiohttp API to check if body exists and can be read
    if not getattr(request, 'body_exists', False) or not getattr(request, 'can_read_body', False):
        return ""
    content = getattr(request, 'content', None)
    if content is None or not hasattr(content, '_buffer'):
        return ""
    buffer = content._buffer
    if not buffer:
        return ""
    try:
        # _buffer is a deque of bytes chunks; join all chunks
        charset = getattr(request, 'charset', None) or 'utf-8'
        raw = b''.join(buffer)
        return raw.decode(charset, errors='replace')[0:MAX_DATA_LENGTH]
    except Exception:
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


def _aiohttp_inject_buffered(response, trace_id) -> None:
    """Append the trace-return trailer + header to a buffered aiohttp web.Response.
    Only web.Response (which has a settable .body) is eligible; web.StreamResponse
    does not have .body and is handled separately via prepare/write_eof hooks."""
    if not hasattr(response, "body"):
        return
    try:
        current = response.body
    except Exception:
        return
    if current is None:
        return
    payload = tr.get_response_trailer(trace_id)
    if payload is None:
        return
    header_value, trailer = payload
    response.body = bytes(current) + trailer
    response.headers[TRACE_RETURN_RESPONSE_HEADER] = header_value


@with_tracer_wrapper
async def aiohttp_streamresponse_prepare(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs):
    """Wrap StreamResponse.prepare to set the trace-return header before headers are
    sent to the client. Does NOT create a span: calls wrapped(...) directly so the
    streaming response's own parent/child span relationships are untouched."""
    try:
        if tr.is_trace_return_enabled() and is_scope_set(TRACE_RETURN_SCOPE_NAME) and not instance.prepared:
            delimiter = tr.make_delimiter()
            instance._monocle_tr_delimiter = delimiter
            instance.headers[TRACE_RETURN_RESPONSE_HEADER] = tr.build_response_header_value(delimiter)
    except Exception as e:
        logger.debug(f"aiohttp stream prepare header skipped: {e}")
    return await wrapped(*args, **kwargs)


@with_tracer_wrapper
async def aiohttp_streamresponse_write_eof(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs):
    """Wrap StreamResponse.write_eof to write the trace-return trailer just before EOF.
    Does NOT create a span: calls wrapped(...) directly (mirrors the prepare wrapper)."""
    try:
        delimiter = getattr(instance, "_monocle_tr_delimiter", None)
        if delimiter is not None:
            trace_id = get_current_monocle_span().get_span_context().trace_id
            trailer = tr.pop_and_build_trailer(trace_id, delimiter)
            if trailer:
                await instance.write(trailer)
            instance._monocle_tr_delimiter = None
    except Exception as e:
        logger.debug(f"aiohttp stream write_eof trailer skipped: {e}")
    return await wrapped(*args, **kwargs)


class aiohttpSpanHandler(HttpSpanHandler):

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return aiohttp_pre_tracing(args), None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        aiohttp_post_tracing(token)

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        return aiohttp_skip_span(args)

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        try:
            # Buffered web.Response only; StreamResponse is handled via the
            # prepare/write_eof hooks registered in methods.py. This runs AFTER
            # hydrate_span (see wrapper.post_process_span), so the response span's
            # data.output has already been captured from the clean, un-trailered
            # result -- mutating result.body here does not pollute the span.
            from aiohttp import web
            if isinstance(result, web.Response):
                trace_id = span.get_span_context().trace_id if span is not None else 0
                _aiohttp_inject_buffered(result, trace_id)
        except Exception as e:
            logger.debug(f"aiohttp trace-return buffered injection skipped: {e}")
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)