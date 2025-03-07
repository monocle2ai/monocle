# pylint: disable=protected-access
import logging
from opentelemetry.trace import Tracer

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import (
    get_fully_qualified_class_name,
    with_tracer_wrapper,
    set_scope,
    remove_scope,
    async_wrapper
)
logger = logging.getLogger(__name__)

def wrapper_processor(async_task: bool, tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, args, kwargs):
    # Some Langchain objects are wrapped elsewhere, so we ignore them here
    if instance.__class__.__name__ in ("AgentExecutor"):
        return wrapped(*args, **kwargs)

    if hasattr(instance, "name") and instance.name:
        name = f"{to_wrap.get('span_name')}.{instance.name.lower()}"
    elif to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = get_fully_qualified_class_name(instance)

    return_value = None
    try:
        handler.pre_tracing(to_wrap, wrapped, instance, args, kwargs)
        if to_wrap.get('skip_span') or handler.skip_span(to_wrap, wrapped, instance, args, kwargs):
            if async_task:
                return_value = async_wrapper(wrapped, None, None, *args, **kwargs)
            else:
                return_value = wrapped(*args, **kwargs)
        else:
            return_value = task_processor(name, async_task, tracer, handler, to_wrap, wrapped, instance, args, kwargs)
        return return_value
    finally:
        handler.post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)

def task_processor(name: str, async_task: bool, tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, args, kwargs):
    # For singleton spans, eg OpenAI inference generate a workflow span to format the workflow specific attributes
    with tracer.start_as_current_span(name) as span:
        token = handler.set_workflow_span(to_wrap, span)
        try:
            if handler.is_non_workflow_root_span(span, to_wrap):
                # This is a singleton span, so call the processor recursively for the actual span
                return_value = task_processor(name, async_task, tracer, handler, to_wrap, wrapped, instance, args, kwargs)
            else:
                token = handler.pre_task_processing(to_wrap, wrapped, instance, args, kwargs, span)
                if async_task:
                    return_value = async_wrapper(wrapped, None, None, *args, **kwargs)
                else:                    return_value = wrapped(*args, **kwargs)
                handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, return_value, span)
                handler.post_task_processing(token, to_wrap, wrapped, instance, args, kwargs, return_value, span)
        finally:
            handler.close_workflow_span(span, token)
    return return_value

@with_tracer_wrapper
def task_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, args, kwargs):
    return wrapper_processor(False, tracer, handler, to_wrap, wrapped, instance, args, kwargs)

@with_tracer_wrapper
async def atask_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, args, kwargs):
    return wrapper_processor(True, tracer, handler, to_wrap, wrapped, instance, args, kwargs)

@with_tracer_wrapper
def scope_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, args, kwargs):
    scope_name = to_wrap.get('scope_name', None)
    if scope_name:
        token = set_scope(scope_name)
    return_value = wrapped(*args, **kwargs)
    if scope_name:
        remove_scope(token)
    return return_value

@with_tracer_wrapper
async def ascope_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, args, kwargs):
    scope_name = to_wrap.get('scope_name', None)
    return_value = async_wrapper(wrapped, scope_name, None, *args, **kwargs)
    return return_value