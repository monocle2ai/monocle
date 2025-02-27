from threading import local
from monocle_apptrace.instrumentation.common.utils import extract_http_headers, clear_http_scopes
from opentelemetry.propagate import extract
from opentelemetry.context import Context, attach, detach
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
token_data = local()
token_data.current_token = None

def flask_pre_processor(to_wrap, wrapped, args):
    headers = dict()
    for key, value in args[0].items():
        if key.startswith("HTTP_"):
            new_key = key[5:].lower().replace("_", "-")
            headers[new_key] = value
    token_data.current_token = extract_http_headers(headers)

def flask_post_processor(to_wrap, wrapped, result, args, kwargs):
    clear_http_scopes(token_data.current_token)
    token_data.current_token = None

class FlaskSpanHandler(SpanHandler):

    def pre_task_processing(self, to_wrap, wrapped, instance, args,kwargs, span):
        flask_pre_processor(to_wrap=to_wrap, wrapped=wrapped, args=args)
        super().pre_task_processing(to_wrap, wrapped, instance, args,kwargs,span)

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, span):
        flask_post_processor(to_wrap=to_wrap, wrapped=wrapped, result=result, args=args, kwargs=kwargs)
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, span)