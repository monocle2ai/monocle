import os
from  monocle_apptrace.instrumentation.metamodel.requests import allowed_urls
from opentelemetry.propagate import inject
from monocle_apptrace.instrumentation.common.utils import get_baggage_for_scopes

def request_pre_processor(args, kwargs):
    # add traceparent to the request headers in kwargs
    if 'url' in kwargs:
        url:str = kwargs['url']
        for allowed_url in allowed_urls:
            if url.startswith(allowed_url):
                if 'headers' not in kwargs:
                    headers = {}
                else:
                    headers = kwargs['headers']
                inject(headers)
                kwargs['headers'] = headers
