import os

from opentelemetry.context import attach, detach, get_value, set_value
from opentelemetry.trace.propagation import _SPAN_KEY
from  monocle_apptrace.instrumentation.metamodel.requests import allowed_urls
from opentelemetry.propagate import inject
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import _MONOCLE_SPAN_KEY, add_monocle_trace_state
from urllib.parse import urlparse, ParseResult


def get_route(kwargs):
    url:str = kwargs['url']
    parsed_url:ParseResult = urlparse(url)
    return f"{parsed_url.netloc}{parsed_url.path}"

def get_method(kwargs) -> str:
    return kwargs['method'] if 'method' in kwargs else 'GET'

def get_params(kwargs) -> dict:
    url:str = kwargs['url']
    parsed_url:ParseResult = urlparse(url)
    return parsed_url.query

def get_headers(kwargs) -> dict:
    return kwargs['headers'] if 'headers' in kwargs else {}

def get_body(kwargs) -> dict:
    body = {}
    return body

def extract_response(result) -> str:
    return result.text if hasattr(result, 'text') else str(result)

def extract_status(result) -> str:
    return f"{result.status_code}"


def request_pre_task_processor(kwargs):
    # add traceparent to the request headers in kwargs
    if 'headers' not in kwargs:
        headers = {}
    else:
        headers = kwargs['headers'].copy()
    add_monocle_trace_state(headers)
    inject(headers)
    kwargs['headers'] = headers

def request_skip_span(kwargs, trace_all_urls: bool) -> bool:
    # add traceparent to the request headers in kwargs
    if trace_all_urls:
        return False
    if 'url' in kwargs:
        url:str = kwargs['url']
        for allowed_url in allowed_urls:
            if url.startswith(allowed_url.strip()):
                return False
    return True

class RequestSpanHandler(SpanHandler):

    _trace_all_urls:bool = False

    @staticmethod
    def set_trace_all_urls_for_test(trace_all:bool):
        RequestSpanHandler._trace_all_urls = trace_all

    def pre_task_processing(self, to_wrap, wrapped, instance, args,kwargs, span):
        token = None
        try:
            token = attach(set_value(_SPAN_KEY, get_value(_MONOCLE_SPAN_KEY)))
            request_pre_task_processor(kwargs)
        finally:
            if token is not None:
                detach(token)
        super().pre_task_processing(to_wrap, wrapped, instance, args,kwargs,span)

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        return request_skip_span(kwargs, RequestSpanHandler._trace_all_urls)