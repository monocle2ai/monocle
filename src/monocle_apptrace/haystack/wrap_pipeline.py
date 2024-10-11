import logging
from opentelemetry import context as context_api
from opentelemetry.context import attach, set_value
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
)
from monocle_apptrace.wrap_common import WORKFLOW_TYPE_MAP, with_tracer_wrapper, CONTEXT_INPUT_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@with_tracer_wrapper
def wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    name = "haystack_pipeline"
    inputs = set()
    workflow_input = get_workflow_input(args, inputs)
    embedding_model = get_embedding_model(instance)

    with tracer.start_as_current_span(f"{name}.workflow") as span:
        ctx = set_value("workflow_name", name)
        ctx = set_value(EMBEDDING_MODEL, embedding_model, ctx)
        ctx = set_value(CONTEXT_INPUT_KEY, workflow_input, ctx)
        attach(ctx)
        workflow_name = span.resource.attributes.get("service.name")
        set_workflow_attributes(span, workflow_name)
        response = wrapped(*args, **kwargs)
    return response

def get_workflow_input(args, inputs):
    for value in args[0].values():
        for text in value.values():
            inputs.add(text)

    workflow_input: str = ""

    for input_str in inputs:
        workflow_input = workflow_input + input_str
    return workflow_input

def set_workflow_attributes(span, workflow_name):
    span.set_attribute("workflow_name",workflow_name)
    span.set_attribute("workflow_type", WORKFLOW_TYPE_MAP["haystack"])

def get_embedding_model(instance):
    try:
        if hasattr(instance, 'get_component'):
            text_embedder = instance.get_component('text_embedder')
            if text_embedder and hasattr(text_embedder, 'model'):
                # Set the embedding model attribute
                return text_embedder.model
    except:
        pass

    return None
