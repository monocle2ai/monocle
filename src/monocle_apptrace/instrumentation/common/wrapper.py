# pylint: disable=protected-access
import logging
from opentelemetry.trace import Tracer
from opentelemetry.context import set_value, attach, detach, get_value

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import (
    get_fully_qualified_class_name,
    with_tracer_wrapper,
    set_scope,
    remove_scope
)
from monocle_apptrace.instrumentation.common.constants import WORKFLOW_TYPE_KEY, ADD_NEW_WORKFLOW
logger = logging.getLogger(__name__)

def get_auto_close_span(to_wrap, kwargs):
    try:
        if to_wrap.get("output_processor") and to_wrap.get("output_processor").get("is_auto_close"):
            return to_wrap.get("output_processor").get("is_auto_close")(kwargs)
        return True
    except Exception as e:
        logger.warning("Warning: Error occurred in get_auto_close_span: %s", str(e))
        return True

def pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path):
    SpanHandler.set_default_monocle_attributes(span, source_path)
    if SpanHandler.is_root_span(span) or add_workflow_span:
        # This is a direct API call of a non-framework type
        SpanHandler.set_workflow_properties(span, to_wrap)
    else:
        SpanHandler.set_non_workflow_properties(span)
        handler.pre_task_processing(to_wrap, wrapped, instance, args, kwargs, span)

def post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, return_value, span):
    if not (SpanHandler.is_root_span(span) or get_value(ADD_NEW_WORKFLOW) == True):
        handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, return_value, span)
        handler.post_task_processing(to_wrap, wrapped, instance, args, kwargs, return_value, span)

def monocle_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
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
            return_value = wrapped(*args, **kwargs)
        else:
            add_workflow_span = get_value(ADD_NEW_WORKFLOW) == True
            token = attach(set_value(ADD_NEW_WORKFLOW, False))
            try:
                # Main span processing logic
                if(get_auto_close_span(to_wrap, kwargs)):
                    with tracer.start_as_current_span(name) as span:
                        pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)
                        
                        if SpanHandler.is_root_span(span) or add_workflow_span:
                            # Recursive call for the actual span
                            return_value = monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)
                        else:
                            with SpanHandler.workflow_type(to_wrap):
                                return_value = wrapped(*args, **kwargs)
                        
                        post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, return_value, span)
                else:
                    span = tracer.start_span(name)
                    
                    pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)
                    
                    def post_process_span_internal(ret_val):
                        nonlocal handler, to_wrap, wrapped, instance, args, kwargs, span
                        post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, ret_val, span)
                        span.end()
                        
                    # if SpanHandler.is_root_span(span) or add_workflow_span:
                    #     # Recursive call for the actual span
                    #     return_value = monocle_wrapper(async_task, tracer, handler, to_wrap, wrapped, instance, args, kwargs)
                    # else:
                    
                    with SpanHandler.workflow_type(to_wrap):
                        return_value = wrapped(*args, **kwargs)
                    if to_wrap.get("output_processor") and to_wrap.get("output_processor").get("response_processor"):
                        # Process the stream
                        to_wrap.get("output_processor").get("response_processor")(to_wrap, return_value, post_process_span_internal)
                    else: 
                        span.end()
                   
            finally:
                detach(token)
        return return_value
    finally:
        handler.post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)
        
async def amonocle_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
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
            return_value = await wrapped(*args, **kwargs)
        else:
            add_workflow_span = get_value(ADD_NEW_WORKFLOW) == True
            token = attach(set_value(ADD_NEW_WORKFLOW, False))
            try:
                # Main span processing logic
                if(get_auto_close_span(to_wrap, kwargs)):
                    with tracer.start_as_current_span(name) as span:
                        pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)
                        
                        if SpanHandler.is_root_span(span) or add_workflow_span:
                            # Recursive call for the actual span
                            return_value = await amonocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)
                        else:
                            with SpanHandler.workflow_type(to_wrap):
                                return_value = await wrapped(*args, **kwargs)
                        
                        post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, return_value, span)
                else:
                    span = tracer.start_span(name)
                    
                    pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)
                    
                    def post_process_span_internal(ret_val):
                        nonlocal handler, to_wrap, wrapped, instance, args, kwargs, span
                        post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, ret_val, span)
                        span.end()
                        
                    # if SpanHandler.is_root_span(span) or add_workflow_span:
                    #     # Recursive call for the actual span
                    #     return_value = monocle_wrapper(async_task, tracer, handler, to_wrap, wrapped, instance, args, kwargs)
                    # else:
                    
                    with SpanHandler.workflow_type(to_wrap):
                        return_value = await wrapped(*args, **kwargs)
                       
                    if to_wrap.get("output_processor") and to_wrap.get("output_processor").get("response_processor"):
                        # Process the stream
                        to_wrap.get("output_processor").get("response_processor")(to_wrap, return_value, post_process_span_internal)
                    else: 
                        span.end()
                   
            finally:
                detach(token)
        return return_value
    finally:
        handler.post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)

@with_tracer_wrapper
def task_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)

@with_tracer_wrapper
async def atask_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return await amonocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)

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
    token = None
    try:
        if scope_name and scope_value:
            token = set_scope(scope_name, scope_value)
        return_value = await wrapped(*args, **kwargs)
        return return_value
    finally:
        if token:
            remove_scope(token)
    
