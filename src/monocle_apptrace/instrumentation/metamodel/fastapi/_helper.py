import logging
from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes, try_option, Option, MonocleSpanException
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import HTTP_SUCCESS_CODES
from urllib.parse import unquote, urlparse, ParseResult
from typing import Dict, Any

logger = logging.getLogger(__name__)
MAX_DATA_LENGTH = 1000
token_data = local()
token_data.current_token = None

def get_url(kwargs) -> ParseResult:
    """Extract URL from FastAPI request."""
    try:
        request = kwargs.get('request')
        if request and hasattr(request, 'url'):
            return urlparse(str(request.url))
        return ParseResult(scheme='', netloc='', path='', params='', query='', fragment='')
    except Exception as e:
        logger.warning(f"Error extracting URL: {e}")
        return ParseResult(scheme='', netloc='', path='', params='', query='', fragment='')

def get_route(kwargs) -> str:
    """Extract route from FastAPI request."""
    try:
        return kwargs[0].get("path")
    except Exception as e:
        logger.warning(f"Error extracting route: {e}")
        return ""

def get_method(kwargs) -> str:
    """Extract HTTP method from FastAPI request."""
    try:
        
        return kwargs[0].get("method")
    except Exception as e:
        logger.warning(f"Error extracting method: {e}")
        return ""

def get_params(args) -> dict:
    """Extract query parameters from FastAPI request."""
    try:
        query_string = args[0].get("query_string", "")
        # query_string = "b'abc=1'"
        return dict(unquote(param).split('=') for param in query_string.split('&') if '=' in param)
    except Exception as e:
        logger.warning(f"Error extracting params: {e}")
        return {}

def get_body(kwargs) -> dict:
    """Extract request body from FastAPI request."""
    try:
        return {}
    except Exception as e:
        logger.warning(f"Error extracting body: {e}")
        return {}

def extract_response(result) -> str:
    """Extract response content from FastAPI response."""
    try:
        return ""
    except Exception as e:
        logger.warning(f"Error extracting response: {e}")
        return ""

def extract_status(result) -> str:
    """Extract status code from FastAPI response."""
    try:
        status = ""
        if hasattr(result, 'status_code'):
            status = str(result.status_code)
        return status
    except MonocleSpanException:
        raise
    except Exception as e:
        logger.warning(f"Error extracting status: {e}")
        return ""

def fastapi_pre_tracing(args):
    """Pre-processing for FastAPI tracing."""
    try:
        request = args[0]
        headers = {}
        if request and request.get('headers'):
            headers = dict(request['headers'])
            # if keys of headers are btytes, decode them
            headers = {k.decode('utf-8') if isinstance(k, bytes) else k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in headers.items()}
        # Extract HTTP headers and set the current token
        token_data.current_token = extract_http_headers(headers)
    except Exception as e:
        logger.warning(f"Error in FastAPI pre-tracing: {e}")
        token_data.current_token = None

def fastapi_post_tracing():
    """Post-processing for FastAPI tracing."""
    try:
        clear_http_scopes(token_data.current_token)
        token_data.current_token = None
    except Exception as e:
        logger.warning(f"Error in FastAPI post-tracing: {e}")

class FastAPISpanHandler(SpanHandler):
    """Custom span handler for FastAPI requests."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        fastapi_pre_tracing(args)
        return super().pre_tracing(to_wrap, wrapped, instance, args, kwargs)
    
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        fastapi_post_tracing()
        return super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)
