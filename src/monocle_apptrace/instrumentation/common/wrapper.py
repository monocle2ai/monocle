# pylint: disable=protected-access
import logging

from opentelemetry.trace import Tracer

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import (
    get_fully_qualified_class_name,
    with_tracer_wrapper,
    set_scope,
    remove_scope
)
from monocle_apptrace.instrumentation.metamodel.botocore import _helper
logger = logging.getLogger(__name__)


@with_tracer_wrapper
def task_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, args, kwargs):

    # Some Langchain objects are wrapped elsewhere, so we ignore them here
    if instance.__class__.__name__ in ("AgentExecutor"):
        return wrapped(*args, **kwargs)

    if hasattr(instance, "name") and instance.name:
        name = f"{to_wrap.get('span_name')}.{instance.name.lower()}"
    elif to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = get_fully_qualified_class_name(instance)

    handler.validate(to_wrap, wrapped, instance, args, kwargs)
    handler.pre_task_action(to_wrap, wrapped, instance, args, kwargs)
    return_value = None
    try:
        if to_wrap.get('skip_span'):
            return_value = wrapped(*args, **kwargs)
        else:
            with tracer.start_as_current_span(name) as span:
                handler.pre_task_processing(to_wrap, wrapped, instance, args, kwargs, span)
                return_value = wrapped(*args, **kwargs)
                handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, return_value, span)
                handler.post_task_processing(to_wrap, wrapped, instance, args, kwargs, return_value, span)
        return return_value
    finally:
        handler.post_task_action(tracer, to_wrap, wrapped, instance, args, kwargs, return_value)


@with_tracer_wrapper
async def atask_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""

    # Some Langchain objects are wrapped elsewhere, so we ignore them here
    if instance.__class__.__name__ in ("AgentExecutor"):
        return wrapped(*args, **kwargs)

    if hasattr(instance, "name") and instance.name:
        name = f"{to_wrap.get('span_name')}.{instance.name.lower()}"
    elif to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = get_fully_qualified_class_name(instance)

    handler.validate(to_wrap, wrapped, instance, args, kwargs)
    handler.pre_task_action(to_wrap, wrapped, instance, args, kwargs)

    try:
        if to_wrap.get('skip_span'):
            return_value = wrapped(*args, **kwargs)
        else:
            with tracer.start_as_current_span(name) as span:
                handler.pre_task_processing(to_wrap, wrapped, instance, args, span)
                return_value = wrapped(*args, **kwargs)
                handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, return_value, span)
                handler.post_task_processing(to_wrap, wrapped, instance, args, kwargs, return_value, span)
        return return_value
    finally:
        handler.post_task_action(tracer, to_wrap, wrapped, instance, return_value, args, kwargs)

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
    if scope_name:
        token = set_scope(scope_name)
    return_value = wrapped(*args, **kwargs)
    if scope_name:
        remove_scope(token)
    return return_value