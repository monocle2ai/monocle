import os
from  monocle_apptrace.instrumentation.metamodel.requests import allowed_urls
from opentelemetry.propagate import inject
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler

def request_pre_processor(to_wrap, wrapped, args, kwargs):
    # add traceparent to the request headers in kwargs
    if 'headers' not in kwargs:
        headers = {}
    else:
        headers = kwargs['headers'].copy()
    inject(headers)
    kwargs['headers'] = headers

def pre_task_action_processor(to_wrap, wrapped, args, kwargs):
    # add traceparent to the request headers in kwargs
    if 'url' in kwargs:
        url:str = kwargs['url']
        for allowed_url in allowed_urls:
            if url.startswith(allowed_url.strip()):
                return
    to_wrap['skip_span'] = True

class RequestSpanHandler(SpanHandler):

    def pre_task_processing(self, to_wrap, wrapped, instance, args,kwargs, span):
        request_pre_processor(to_wrap=to_wrap, wrapped=wrapped, args=args,kwargs=kwargs)
        pre_task_action_processor(to_wrap, wrapped, args, kwargs)
        super().pre_task_processing(to_wrap, wrapped, instance, args,kwargs,span)
