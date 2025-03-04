import os
from  monocle_apptrace.instrumentation.metamodel.requests import allowed_urls
from opentelemetry.propagate import inject
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler

def request_pre_task_processor(kwargs):
    # add traceparent to the request headers in kwargs
    if 'headers' not in kwargs:
        headers = {}
    else:
        headers = kwargs['headers'].copy()
    inject(headers)
    kwargs['headers'] = headers

def request_skip_span(kwargs) -> bool:
    # add traceparent to the request headers in kwargs
    if 'url' in kwargs:
        url:str = kwargs['url']
        for allowed_url in allowed_urls:
            if url.startswith(allowed_url.strip()):
                return False
    return True

class RequestSpanHandler(SpanHandler):

    def pre_task_processing(self, to_wrap, wrapped, instance, args,kwargs, span):
        request_pre_task_processor(kwargs)
        super().pre_task_processing(to_wrap, wrapped, instance, args,kwargs,span)

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        return request_skip_span(kwargs)