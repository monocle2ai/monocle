import os
from  monocle_apptrace.instrumentation.metamodel.requests import allowed_urls
from opentelemetry.propagate import inject

def request_pre_processor(to_wrap, wrapped, result, args, kwargs):
    # add traceparent to the request headers in kwargs
    if 'headers' not in kwargs:
        headers = {}
    else:
        headers = kwargs['headers'].copy()
    inject(headers)
    kwargs['headers'] = headers

def pre_task_action_processor(to_wrap, wrapped, result, args, kwargs):
    # add traceparent to the request headers in kwargs
    if 'url' in kwargs:
        url:str = kwargs['url']
        for allowed_url in allowed_urls:
            if url.startswith(allowed_url):
                return
    to_wrap['skip_span'] = True