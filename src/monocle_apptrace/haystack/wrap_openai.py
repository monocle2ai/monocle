import logging
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from monocle_apptrace.wrap_common import with_tracer_wrapper
from monocle_apptrace.utils import (
    dont_throw,
    set_span_attribute
)

logger = logging.getLogger(__name__)

@dont_throw
def _set_input_attributes(span, kwargs, instance, args):
    set_span_attribute(span, "llm_input", kwargs.get("prompt"))

    if 'model' in instance.__dict__:
        model_name = instance.__dict__.get("model")
        set_span_attribute(span, "model_name", model_name)

@dont_throw
def _set_response_attributes(span, response):

    if "meta" in response:
        token_usage = response["meta"][0]["usage"]
        set_span_attribute(span, "completion_tokens", token_usage.get("completion_tokens"))
        set_span_attribute(span, "prompt_tokens", token_usage.get("prompt_tokens"))
        set_span_attribute(span, "total_tokens", token_usage.get("total_tokens"))


@with_tracer_wrapper
def wrap_openai(tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    with tracer.start_as_current_span("haystack.openai") as span:
        if span.is_recording():
            _set_input_attributes(span, kwargs, instance, args)
        response = wrapped(*args, **kwargs)

        if response:
            if span.is_recording():
                _set_response_attributes(span, response)

        return response
