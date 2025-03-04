from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes
from opentelemetry.propagate import extract
from opentelemetry.context import Context, attach, detach
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
token_data = local()
token_data.current_token = None

def flask_pre_tracing(args):
    headers = dict()
    for key, value in args[0].items():
        if key.startswith("HTTP_"):
            new_key = key[5:].lower().replace("_", "-")
            headers[new_key] = value
    token_data.current_token = extract_http_headers(headers)

def flask_post_tracing():
    clear_http_scopes(token_data.current_token)
    token_data.current_token = None

class FlaskSpanHandler(SpanHandler):

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        flask_pre_tracing(args)
        return super().pre_tracing(to_wrap, wrapped, instance, args, kwargs)
    
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        flask_post_tracing()
        return super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)