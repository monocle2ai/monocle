from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes
from opentelemetry.propagate import extract
from opentelemetry.context import Context, attach, detach

token_data = local()
token_data.current_token = None

def flask_pre_processor(args, kwargs):
    headers = dict()
    for key, value in args[0].items():
        if key.startswith("HTTP_"):
            new_key = key[5:].lower().replace("_", "-")
            headers[new_key] = value
    token_data.current_token = extract_http_headers(headers)

def flask_post_processor(tracer, to_wrap, wrapped, instance, args, kwargs,return_value):
    clear_http_scopes(token_data.current_token)
    token_data.current_token = None
