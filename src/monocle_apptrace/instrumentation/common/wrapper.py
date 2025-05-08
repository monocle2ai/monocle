# pylint: disable=protected-access
import logging
from opentelemetry.trace import Tracer
from opentelemetry.context import set_value, attach, detach, get_value

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import (
    get_fully_qualified_class_name,
    with_tracer_wrapper,
    set_scope,
    remove_scope,
    async_wrapper
)
from monocle_apptrace.instrumentation.common.constants import WORKFLOW_TYPE_KEY, ADD_NEW_WORKFLOW
logger = logging.getLogger(__name__)

def wrapper_processor(async_task: bool, tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    # Some Langchain objects are wrapped elsewhere, so we ignore them here
    if instance.__class__.__name__ in ("AgentExecutor"):
        return wrapped(*args, **kwargs)

    if to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = get_fully_qualified_class_name(instance)

    return_value = None
    token = None
    try:
        handler.pre_tracing(to_wrap, wrapped, instance, args, kwargs)
        if to_wrap.get('skip_span', False) or handler.skip_span(to_wrap, wrapped, instance, args, kwargs):
            if async_task:
                return_value = async_wrapper(wrapped, None, None, None, *args, **kwargs)
            else:
                return_value = wrapped(*args, **kwargs)
        else:
            add_workflow_span = get_value(ADD_NEW_WORKFLOW) == True
            token = attach(set_value(ADD_NEW_WORKFLOW, False))
            try:
                return_value = span_processor(name, async_task, tracer, handler, add_workflow_span,
                                        to_wrap, wrapped, instance, source_path, args, kwargs)
            finally:
                detach(token)
        return return_value
    finally:
        handler.post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)

def span_processor(name: str, async_task: bool, tracer: Tracer, handler: SpanHandler, add_workflow_span:bool,
                to_wrap, wrapped, instance, source_path, args, kwargs):
    # For singleton spans, eg OpenAI inference generate a workflow span to format the workflow specific attributes
    return_value = None
    with tracer.start_as_current_span(name) as span:
        # Since Spanhandler can be overridden, ensure we set default monocle attributes.
        SpanHandler.set_default_monocle_attributes(span, source_path)
        if SpanHandler.is_root_span(span) or add_workflow_span:
            # This is a direct API call of a non-framework type, call the span_processor recursively for the actual span
            SpanHandler.set_workflow_properties(span, to_wrap)
            return_value = span_processor(name, async_task, tracer, handler, False, to_wrap, wrapped, instance, source_path, args, kwargs)
        else:
            with SpanHandler.workflow_type(to_wrap):
                SpanHandler.set_non_workflow_properties(span)
                handler.pre_task_processing(to_wrap, wrapped, instance, args, kwargs, span)
                if async_task:
                    return_value = async_wrapper(wrapped, None, None, None, *args, **kwargs)
                else:                    
                    return_value = wrapped(*args, **kwargs)
                handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, return_value, span)
                handler.post_task_processing(to_wrap, wrapped, instance, args, kwargs, return_value, span)
    return return_value

@with_tracer_wrapper
def task_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return wrapper_processor(False, tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)

@with_tracer_wrapper
async def atask_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return wrapper_processor(True, tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)

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
    scope_value = to_wrap.get('scope_value', None)
    return_value = async_wrapper(wrapped, scope_name, scope_value, None, *args, **kwargs)
    return return_value