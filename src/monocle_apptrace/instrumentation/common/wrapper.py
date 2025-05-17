# pylint: disable=protected-access
import logging
from opentelemetry.trace import Tracer
from opentelemetry.context import set_value, attach, detach, get_value

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import (
    get_fully_qualified_class_name,
    set_scopes,
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
        try:
            handler.pre_task_processing(to_wrap, wrapped, instance, args, kwargs, span)
        except Exception as e:
            logger.info(f"Warning: Error occurred in pre_task_processing: {e}")

def post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, return_value, span):
    if not (SpanHandler.is_root_span(span) or get_value(ADD_NEW_WORKFLOW) == True):
        try:
            handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, return_value, span)
        except Exception as e:
            logger.info(f"Warning: Error occurred in hydrate_span: {e}")
        try:
            handler.post_task_processing(to_wrap, wrapped, instance, args, kwargs, return_value, span)
        except Exception as e:
            logger.info(f"Warning: Error occurred in post_task_processing: {e}")

def get_span_name(to_wrap, instance):
    if to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = get_fully_qualified_class_name(instance)
    return name

def monocle_wrapper_span_processor(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, add_workflow_span, args, kwargs):
    # Main span processing logic
    name = get_span_name(to_wrap, instance)
    return_value = None
    span_status = None
    if(get_auto_close_span(to_wrap, kwargs)):
        with tracer.start_as_current_span(name) as span:
            pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)
            
            if SpanHandler.is_root_span(span) or add_workflow_span:
                # Recursive call for the actual span
                return_value, span_status = monocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, False, args, kwargs)
                span.set_status(span_status)
            else:
                try:
                    with SpanHandler.workflow_type(to_wrap, span):
                        return_value = wrapped(*args, **kwargs)
                finally:
                    post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, return_value, span)
                span_status = span.status
    else:
        span = tracer.start_span(name)
        
        pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)
        
        def post_process_span_internal(ret_val):
            nonlocal handler, to_wrap, wrapped, instance, args, kwargs, span
            post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, ret_val, span)
            span.end()
                    
        try:
            with SpanHandler.workflow_type(to_wrap, span):
                return_value = wrapped(*args, **kwargs)
        finally:
            if to_wrap.get("output_processor") and to_wrap.get("output_processor").get("response_processor"):
                # Process the stream
                to_wrap.get("output_processor").get("response_processor")(to_wrap, return_value, post_process_span_internal)
            else: 
                span.end()
        span_status = span.status
    return return_value, span_status

def monocle_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return_value = None
    token = None
    try:
        try:
            handler.pre_tracing(to_wrap, wrapped, instance, args, kwargs)
        except Exception as e:
            logger.info(f"Warning: Error occurred in pre_tracing: {e}")
        if to_wrap.get('skip_span', False) or handler.skip_span(to_wrap, wrapped, instance, args, kwargs):
            return_value = wrapped(*args, **kwargs)
        else:
            add_workflow_span = get_value(ADD_NEW_WORKFLOW) == True
            token = attach(set_value(ADD_NEW_WORKFLOW, False))
            try:
                return_value, span_status = monocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, add_workflow_span, args, kwargs)
            finally:
                detach(token)
        return return_value
    finally:
        try:
            handler.post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)
        except Exception as e:
            logger.info(f"Warning: Error occurred in post_tracing: {e}")

async def amonocle_wrapper_span_processor(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, add_workflow_span, args, kwargs):
    # Main span processing logic
    name = get_span_name(to_wrap, instance)
    return_value = None
    span_status = None
    if(get_auto_close_span(to_wrap, kwargs)):
        with tracer.start_as_current_span(name) as span:
            pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)
            
            if SpanHandler.is_root_span(span) or add_workflow_span:
                # Recursive call for the actual span
                return_value, span_status = await amonocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, False, args, kwargs)
                span.set_status(span_status)
            else:
                try:
                    with SpanHandler.workflow_type(to_wrap, span):
                        return_value = await wrapped(*args, **kwargs)
                finally:
                    post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, return_value, span)
                span_status = span.status
    else:
        span = tracer.start_span(name)
        
        pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)
        
        def post_process_span_internal(ret_val):
            nonlocal handler, to_wrap, wrapped, instance, args, kwargs, span
            post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, ret_val, span)
            span.end()

        try:
            with SpanHandler.workflow_type(to_wrap, span):
                return_value = await wrapped(*args, **kwargs)
        finally:
            if to_wrap.get("output_processor") and to_wrap.get("output_processor").get("response_processor"):
                # Process the stream
                to_wrap.get("output_processor").get("response_processor")(to_wrap, return_value, post_process_span_internal)
            else: 
                span.end()
        span_status = span.status
    return return_value, span.status

async def amonocle_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return_value = None
    token = None
    try:
        try:
            handler.pre_tracing(to_wrap, wrapped, instance, args, kwargs)
        except Exception as e:
            logger.info(f"Warning: Error occurred in pre_tracing: {e}")
        if to_wrap.get('skip_span', False) or handler.skip_span(to_wrap, wrapped, instance, args, kwargs):
            return_value = await wrapped(*args, **kwargs)
        else:
            add_workflow_span = get_value(ADD_NEW_WORKFLOW) == True
            token = attach(set_value(ADD_NEW_WORKFLOW, False))
            try:
                return_value, span_status = await amonocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, add_workflow_span, args, kwargs)    
            finally:
                detach(token)
        return return_value
    finally:
        try:
            handler.post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)
        except Exception as e:
            logger.info(f"Warning: Error occurred in post_tracing: {e}")

@with_tracer_wrapper
def task_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)

@with_tracer_wrapper
async def atask_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return await amonocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)

@with_tracer_wrapper
def scope_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    scope_name = to_wrap.get('scope_name', None)
    if scope_name:
        token = set_scope(scope_name)
    return_value = wrapped(*args, **kwargs)
    if scope_name:
        remove_scope(token)
    return return_value

@with_tracer_wrapper
async def ascope_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    scope_name = to_wrap.get('scope_name', None)
    scope_value = to_wrap.get('scope_value', None)
    token = None
    try:
        if scope_name:
            token = set_scope(scope_name, scope_value)
        return_value = await wrapped(*args, **kwargs)
        return return_value
    finally:
        if token:
            remove_scope(token)
            
@with_tracer_wrapper
def scopes_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    scope_values = to_wrap.get('scope_values', None)
    scope_values = evaluate_scope_values(args, kwargs, scope_values)
    token = None
    try:
        if scope_values:
            token = set_scopes(scope_values)
        return_value = wrapped(*args, **kwargs)
        return return_value
    finally:
        if token:
            remove_scope(token)

@with_tracer_wrapper
async def ascopes_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    scope_values = to_wrap.get('scope_values', None)
    scope_values = evaluate_scope_values(args, kwargs, scope_values)
    token = None
    try:
        if scope_values:
            token = set_scopes(scope_values)
        return_value = await wrapped(*args, **kwargs)
        return return_value
    finally:
        if token:
            remove_scope(token)

def evaluate_scope_values(args, kwargs, scope_values):
    if callable(scope_values):
        try:
            scope_values = scope_values(args, kwargs)
        except Exception as e:
            logger.warning("Warning: Error occurred in evaluate_scope_values: %s", str(e))
            scope_values = None
    if isinstance(scope_values, dict):
        return scope_values
    return None
    
