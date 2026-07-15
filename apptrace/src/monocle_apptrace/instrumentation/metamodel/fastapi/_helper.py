import json
import logging
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, List

from opentelemetry.trace import Span

from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from monocle_apptrace.instrumentation.common.span_handler import HttpSpanHandler, SpanHandler
from monocle_apptrace.instrumentation.common.utils import (
    MonocleSpanException,
    clear_http_scopes,
    extract_http_headers,
    get_exception_status_code,
    with_tracer_wrapper,
)
from monocle_apptrace.instrumentation.common.wrapper import amonocle_wrapper

logger = logging.getLogger(__name__)

MAX_DATA_LENGTH = 1000
MAX_STREAMING_CAPTURE_LENGTH = 5000  # Larger buffer for streaming responses

# STREAMING UTILITIES

@dataclass
class StreamingState:
    """State for tracking streaming response capture."""
    chunks: List[bytes] = field(default_factory=list)
    total_length: int = 0
    max_length: int = MAX_STREAMING_CAPTURE_LENGTH
    is_complete: bool = False

    def add_chunk(self, chunk: bytes) -> None:
        """Add a chunk to the captured content."""
        if self.total_length < self.max_length:
            self.chunks.append(chunk)
            self.total_length += len(chunk)

    def get_captured_content(self) -> str:
        """Get the captured content as a string."""
        try:
            content = b"".join(self.chunks)
            decoded = content.decode('utf-8')
            if self.total_length >= self.max_length:
                return decoded[:self.max_length] + "...[truncated]"
            return decoded
        except Exception as e:
            logger.warning(f"Error decoding captured stream: {e}")
            return ""


def _wrap_body_iterator(body_iterator: AsyncIterator, state: StreamingState) -> AsyncIterator:
    """Wrap an async body_iterator to capture chunks while yielding to client."""
    async def capturing_iterator():
        try:
            async for chunk in body_iterator:
                if isinstance(chunk, bytes):
                    state.add_chunk(chunk)
                elif isinstance(chunk, str):
                    state.add_chunk(chunk.encode('utf-8'))
                yield chunk
        finally:
            state.is_complete = True
    return capturing_iterator()


def _extract_sse_content(raw_content: str) -> str:
    """Extract and summarize SSE (Server-Sent Events) content.

    SSE format: 'data: {...}\\n\\n' for each event.
    This extracts the JSON payloads and accumulates delta content.
    """
    try:
        events = []
        accumulated_text = []

        for line in raw_content.split('\n'):
            line = line.strip()
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                try:
                    parsed = json.loads(data)
                    events.append(parsed)
                    # Accumulate delta content (common in LLM streaming responses)
                    if isinstance(parsed, dict):
                        delta = parsed.get('delta') or parsed.get('content') or parsed.get('text')
                        if delta and isinstance(delta, str):
                            accumulated_text.append(delta)
                except json.JSONDecodeError:
                    events.append(data)

        if not events:
            return raw_content[:MAX_STREAMING_CAPTURE_LENGTH]

        # Collect event types for summary
        event_types = set()
        for e in events:
            if isinstance(e, dict) and 'type' in e:
                event_types.add(e['type'])

        result = {
            "_sse_events": len(events),
            "_event_types": list(event_types) if event_types else None,
        }

        if accumulated_text:
            result["response"] = "".join(accumulated_text)

        if events:
            result["first_event"] = events[0]
            if len(events) > 1:
                result["last_event"] = events[-1]

        return json.dumps(result)[:MAX_STREAMING_CAPTURE_LENGTH]
    except Exception as e:
        logger.warning(f"Error extracting SSE content: {e}")
        return raw_content[:MAX_STREAMING_CAPTURE_LENGTH]

# ASGI MIDDLEWARE HELPERS
async def _buffer_request_body(scope, receive):
    """Buffer and replay request body for tracing.

    Consumes the ASGI receive stream, stores the body in scope['_request_body'],
    and returns a new receive callable that replays the buffered messages.
    """
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


async def _capture_response_body(scope, send):
    """Wrap ASGI send to capture response body chunks and status code.

    Captures both regular and streaming response bodies.
    Stores captured content in scope['_response_body'] and status in scope['_response_status'].
    """
    body_chunks = []
    total_length = 0

    async def _send(message):
        nonlocal total_length

        if message.get("type") == "http.response.start":
            scope["_response_status"] = message.get("status", 0)

        if message.get("type") == "http.response.body":
            body = message.get("body", b"")
            if body and total_length < MAX_STREAMING_CAPTURE_LENGTH:
                body_chunks.append(body)
                total_length += len(body)

        scope["_response_body"] = b"".join(body_chunks)
        await send(message)

    return _send

def get_url(args) -> str:
    """Extract full URL from ASGI scope."""
    server = args.get('server', ('127.0.0.1', 80))
    host, port = server
    path = args.get('path', '/')
    scheme = args.get('scheme', 'http')
    return f"{scheme}://{host}:{port}{path}"


def get_route(scope) -> str:
    """Extract route pattern from ASGI scope."""
    route = scope.get('route')
    if route is not None and getattr(route, 'path', None):
        return route.path
    return scope.get('path', '')


def get_method(scope) -> str:
    """Extract HTTP method from ASGI scope."""
    return scope.get('method', '')


def get_params(args) -> str:
    """Extract query string parameters from ASGI scope."""
    try:
        query_bytes = args.get("query_string", b"")
        if not query_bytes:
            return ""
        query_str = query_bytes.decode('utf-8') if isinstance(query_bytes, bytes) else str(query_bytes)
        return urllib.parse.unquote(query_str) if query_str else ""
    except Exception as e:
        logger.warning(f"Error extracting params: {e}")
        return ""


def get_body(args) -> str:
    """Extract request body from scope (buffered by _buffer_request_body)."""
    try:
        body = args.get('_request_body', b'')
        if not body:
            return ""
        decoded = body.decode('utf-8') if isinstance(body, bytes) else str(body)
        try:
            parsed = json.loads(decoded)
            return json.dumps(parsed)[:MAX_DATA_LENGTH]
        except (json.JSONDecodeError, TypeError):
            return decoded[:MAX_DATA_LENGTH]
    except Exception as e:
        logger.warning(f"Error extracting body: {e}")
        return ""


def get_response_from_scope(scope) -> str:
    """Extract response body from scope (captured by _capture_response_body).

    For streaming responses (like SSE), parses and summarizes the content.
    For regular responses, returns the JSON body.
    """
    try:
        response_body = scope.get('_response_body', b'')
        if not response_body:
            return ""

        decoded = response_body.decode('utf-8') if isinstance(response_body, bytes) else str(response_body)

        # Check if it's SSE format (text/event-stream)
        if decoded.startswith('data: '):
            return _extract_sse_content(decoded)

        # Try to parse as JSON for regular responses
        try:
            parsed = json.loads(decoded)
            # Treat empty containers (e.g. health-check responses that return {})
            # as no response so they are not recorded and can be sampled out.
            if isinstance(parsed, (dict, list)) and not parsed:
                return ""
            return json.dumps(parsed)[:MAX_DATA_LENGTH]
        except (json.JSONDecodeError, TypeError):
            return decoded[:MAX_DATA_LENGTH]
    except Exception as e:
        logger.warning(f"Error extracting response from scope: {e}")
        return ""


def get_status_from_scope(scope) -> str:
    """Extract HTTP status code from scope (captured by _capture_response_body)."""
    status = scope.get('_response_status', 0)
    return str(status) if status else ""


def extract_response(response) -> str:
    """Extract response body from regular or streaming responses."""
    try:
        # Handle StreamingResponse with captured content from our wrapper
        if hasattr(response, '_monocle_streaming_state'):
            state = response._monocle_streaming_state
            captured = state.get_captured_content()
            if captured:
                # For SSE, extract just the data payloads for cleaner output
                if getattr(response, 'media_type', '') == 'text/event-stream':
                    return _extract_sse_content(captured)
                return captured[:MAX_STREAMING_CAPTURE_LENGTH]
            # Fallback to metadata if capture failed
            media_type = getattr(response, 'media_type', 'unknown')
            status = getattr(response, 'status_code', 'unknown')
            return json.dumps({
                "_streaming": True,
                "_capture_failed": True,
                "media_type": media_type,
                "status_code": status
            })
        
        # Handle StreamingResponse without capture (shouldn't happen with our wrapper)
        if hasattr(response, 'body_iterator') and not hasattr(response, 'body'):
            media_type = getattr(response, 'media_type', 'unknown')
            status = getattr(response, 'status_code', 'unknown')
            return json.dumps({
                "_streaming": True,
                "media_type": media_type,
                "status_code": status
            })
        
        # Handle regular Response with body
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
    """Extract HTTP status code from the response instance.

    Raises MonocleSpanException for non-success status codes.
    """
    if arguments["exception"] is not None:
        return get_exception_status_code(arguments)

    instance = arguments['instance']
    if hasattr(instance, 'status_code'):
        status = f"{instance.status_code}"
        if status not in HTTP_SUCCESS_CODES:
            is_streaming = hasattr(instance, 'body_iterator') and not hasattr(instance, 'body')
            if is_streaming:
                error_message = f"streaming response with status {status}"
            else:
                error_message = extract_response(instance)
            raise MonocleSpanException(f"error: {status} - {error_message}", status)
        return status
    return "Unknown"

@with_tracer_wrapper
async def fastapi_atask_wrapper(tracer, handler, to_wrap, wrapped, instance,
                                source_path, args, kwargs):
    """Wrap APIRoute.handle to capture request/response bodies.

    - Captures POST body into scope['_request_body']
    - Captures response body (including streaming) into scope['_response_body']

    This approach captures streaming response content WITHOUT creating a separate
    span, preserving correct parent-child span relationships.
    """
    scope, receive, send = args[0], args[1], args[2]

    # Buffer request body for POST/PUT/PATCH
    if scope.get('method', 'GET') in ('POST', 'PUT', 'PATCH'):
        receive = await _buffer_request_body(scope, receive)

    # Wrap send to capture response body (including streaming)
    send = await _capture_response_body(scope, send)

    # Rebuild args with wrapped receive and send
    args = (scope, receive, send) + args[3:]

    return await amonocle_wrapper(
        tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs
    )


@with_tracer_wrapper
async def streaming_response_wrapper(tracer, handler, to_wrap, wrapped, instance,
                                     source_path, args, kwargs):
    """Wrap StreamingResponse.__call__ to capture streaming body content.

    IMPORTANT: This wrapper does NOT create a new span context via amonocle_wrapper.
    It only wraps the body_iterator to capture content. Creating a span here would
    break parent-child relationships since the actual work (agent execution) happens
    during iteration.

    The captured content is attached to the instance for later extraction by
    extract_response() when the parent span's events are hydrated.
    """
    streaming_state = StreamingState()
    instance._monocle_streaming_state = streaming_state

    if hasattr(instance, 'body_iterator'):
        original_iterator = instance.body_iterator
        instance.body_iterator = _wrap_body_iterator(original_iterator, streaming_state)

    # Call the original __call__ directly - no span wrapper
    # This preserves the existing span context so child spans parent correctly
    return await wrapped(*args, **kwargs)

def fastapi_pre_tracing(scope):
    """Extract HTTP headers from ASGI scope for trace context propagation."""
    headers = {
        k.decode('utf-8').lower(): v.decode('utf-8')
        for k, v in scope.get('headers', [])
    }
    return extract_http_headers(headers)


def fastapi_post_tracing(token):
    """Clean up HTTP scopes after request processing."""
    clear_http_scopes(token)

class FastAPISpanHandler(HttpSpanHandler):
    """Span handler for FastAPI HTTP requests."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        scope = args[0] if args else {}
        return fastapi_pre_tracing(scope), None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token):
        fastapi_post_tracing(token)


class FastAPIResponseSpanHandler(SpanHandler):
    """Span handler for FastAPI response data collection.

    This span is only used to collect the data.input and data.output events
    and merge with parent span. It's never exported by itself.
    """

    def should_sample(self, to_wrap, wrapped, instance, args, kwargs, result, ex,
                      span: Span, parent_span: Span) -> bool:
        return False

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex,
                             span: Span, parent_span: Span):
        try:
            if parent_span is not None:
                self.hydrate_events(to_wrap, wrapped, instance, args, kwargs,
                                    result, span=parent_span, is_post_exec=False)
                self.hydrate_events(to_wrap, wrapped, instance, args, kwargs,
                                    result, span=parent_span, is_post_exec=True)
        except Exception as e:
            logger.info(f"Failed to propagate fastapi response: {e}")
        super().post_task_processing(
            to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span
        )
