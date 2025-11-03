# pylint: disable=protected-access
import logging
import os
from contextlib import contextmanager
import os
from typing import AsyncGenerator, Iterator
import logging
from opentelemetry.trace import Tracer
from opentelemetry.trace.propagation import set_span_in_context, get_current_span
from opentelemetry.context import set_value, attach, detach, get_value
from opentelemetry.trace.span import INVALID_SPAN, Span
from opentelemetry.trace.status import StatusCode

from monocle_apptrace.instrumentation.common.constants import (
    ADD_NEW_WORKFLOW,
    AGENTIC_SPANS,
    WORKFLOW_TYPE_KEY,
)
from monocle_apptrace.instrumentation.common.scope_wrapper import monocle_trace_scope
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import (
    get_current_monocle_span,
    remove_scope,
    set_monocle_span_in_context,
    set_scope,
    set_scopes,
    with_tracer_wrapper,
    set_scope,
    remove_scope,
    get_current_monocle_span,
    set_monocle_span_in_context,
)

logger = logging.getLogger(__name__)
ISOLATE_MONOCLE_SPANS = os.getenv("MONOCLE_ISOLATE_SPANS", "true").lower() == "true"

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

def post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, return_value, span, parent_span, ex):
    if not (SpanHandler.is_root_span(span) or get_value(ADD_NEW_WORKFLOW) == True):
        try:
            if parent_span == INVALID_SPAN:
                parent_span = None
            handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, return_value, span, parent_span, ex,
                    is_post_exec=True)
            return_value = SpanHandler.replace_placeholders_in_response(return_value, span)
        except Exception as e:
            logger.info(f"Warning: Error occurred in hydrate_span: {e}")
        
        try:
            handler.post_task_processing(to_wrap, wrapped, instance, args, kwargs, return_value, ex, span, parent_span)
        except Exception as e:
            logger.info(f"Warning: Error occurred in post_task_processing: {e}")
        return return_value

def get_span_name(to_wrap, instance):
    if to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = to_wrap.get("package", "") + "." + to_wrap.get("object", "") + "." + to_wrap.get("method", "")
    return name

def monocle_wrapper_span_processor(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, add_workflow_span, args, kwargs):
    # Main span processing logic
    name = get_span_name(to_wrap, instance)
    return_value = None
    span_status = None
    auto_close_span = get_auto_close_span(to_wrap, kwargs)
    parent_span = get_current_monocle_span()
    with start_as_monocle_span(tracer, name, auto_close_span) as span:
        pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)
        
        if SpanHandler.is_root_span(span) or add_workflow_span:
            # Recursive call for the actual span
            return_value, span_status = monocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, False, args, kwargs)
            span.set_status(StatusCode.OK)
            if not auto_close_span:
                span.end()
        else:
            ex:Exception = None
            to_wrap = get_wrapper_with_next_processor(to_wrap, handler, instance, span, parent_span, args, kwargs)
            if has_more_processors(to_wrap):
                try:
                    handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, None, span, parent_span, ex,
                                is_post_exec=False)
                except Exception as e:
                    logger.info(f"Warning: Error occurred in hydrate_span pre_process_span: {e}")
                try:
                    with monocle_trace_scope(get_builtin_scope_names(to_wrap)):
                        return_value, span_status = monocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, False, args, kwargs)
                except Exception as e:
                    ex = e
                    raise
                finally:
                    return_value = post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, return_value, span, parent_span ,ex)
            else:
                try:
                    handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, None, span, parent_span, ex,
                                is_post_exec=False)
                except Exception as e:
                    logger.info(f"Warning: Error occurred in hydrate_span pre_process_span: {e}")
                try:
                    skip_execution, return_value = SpanHandler.skip_execution(span)
                    if not skip_execution:
                        with SpanHandler.workflow_type(to_wrap, span):
                            return_value = wrapped(*args, **kwargs)
                except Exception as e:
                    ex = e
                    raise
                finally:
                    def post_process_span_internal(ret_val):
                        post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, ret_val, span, parent_span ,ex)
                        if not auto_close_span:
                            span.end()
                        return ret_val
                    if ex is None and not auto_close_span and to_wrap.get("output_processor") and to_wrap.get("output_processor").get("response_processor"):
                        to_wrap.get("output_processor").get("response_processor")(to_wrap, return_value, post_process_span_internal)
                    else:
                        return_value = post_process_span_internal(return_value)
            span_status = span.status
    return return_value, span_status

def monocle_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return_value = None
    pre_trace_token = None
    token = None
    try:
        try:
            pre_trace_token = handler.pre_tracing(to_wrap, wrapped, instance, args, kwargs)
        except Exception as e:
            logger.info(f"Warning: Error occurred in pre_tracing: {e}")
        if to_wrap.get('skip_span', False) or handler.skip_span(to_wrap, wrapped, instance, args, kwargs):
            return_value = wrapped(*args, **kwargs)
        else:
            add_workflow_span = get_value(ADD_NEW_WORKFLOW) == True
            token = attach(set_value(ADD_NEW_WORKFLOW, False))
            try:
                with monocle_trace_scope(get_builtin_scope_names(to_wrap)):
                    return_value, span_status = monocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, add_workflow_span, args, kwargs)
            finally:
                detach(token)
        return return_value
    finally:
        try:
            handler.post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value, token=pre_trace_token)
        except Exception as e:
            logger.info(f"Warning: Error occurred in post_tracing: {e}")

async def amonocle_wrapper_span_processor(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, add_workflow_span,
                                        args, kwargs):
    # Main span processing logic
    name = get_span_name(to_wrap, instance)
    return_value = None
    span_status = None
    auto_close_span = get_auto_close_span(to_wrap, kwargs)
    parent_span = get_current_monocle_span()
    with start_as_monocle_span(tracer, name, auto_close_span) as span:
        pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)
        
        if SpanHandler.is_root_span(span) or add_workflow_span:
            # Recursive call for the actual span
            return_value, span_status = await amonocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, False, args, kwargs)
            span.set_status(StatusCode.OK)
            if not auto_close_span:
                span.end()
        else:
            ex:Exception = None
            to_wrap = get_wrapper_with_next_processor(to_wrap, handler, instance, span, parent_span,args, kwargs)
            if has_more_processors(to_wrap):
                try:
                    handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, None, span, parent_span, ex,
                                    is_post_exec=False)
                except Exception as e:
                    logger.info(f"Warning: Error occurred in hydrate_span pre_process_span: {e}")

                try:
                    with monocle_trace_scope(get_builtin_scope_names(to_wrap)):
                        return_value, span_status = await amonocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, False, args, kwargs)
                except Exception as e:
                    ex = e
                    raise
                finally:
                    return_value = post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, return_value, span, parent_span ,ex)
            else:
                try:
                    handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, None, span, parent_span, ex,
                                is_post_exec=False)
                except Exception as e:
                    logger.info(f"Warning: Error occurred in hydrate_span pre_process_span: {e}")
                try:
                    skip_execution, return_value = SpanHandler.skip_execution(span)
                    if not skip_execution:
                        with SpanHandler.workflow_type(to_wrap, span):
                            return_value = await wrapped(*args, **kwargs)
                except Exception as e:
                    ex = e
                    raise
                finally:
                    def post_process_span_internal(ret_val):
                        ret_val = post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, ret_val, span, parent_span, ex)
                        if not auto_close_span:
                            span.end()
                        return ret_val
                    if ex is None and not auto_close_span and to_wrap.get("output_processor") and to_wrap.get("output_processor").get("response_processor"):
                        to_wrap.get("output_processor").get("response_processor")(to_wrap, return_value, post_process_span_internal)
                    else:
                        return_value = post_process_span_internal(return_value)
        span_status = span.status
    return return_value, span_status

async def amonocle_iter_wrapper_span_processor(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, add_workflow_span,
                                        args, kwargs) -> AsyncGenerator[any, None]:
    # Main span processing logic
    name = get_span_name(to_wrap, instance)
    auto_close_span = get_auto_close_span(to_wrap, kwargs)
    parent_span = get_current_monocle_span()
    last_item = None

    with start_as_monocle_span(tracer, name, auto_close_span) as span:
        pre_process_span(name, tracer, handler, add_workflow_span, to_wrap, wrapped, instance, args, kwargs, span, source_path)

        if SpanHandler.is_root_span(span) or add_workflow_span:
            # Recursive call for the actual span
            async for item in amonocle_iter_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, False, args, kwargs):
                yield item
            span.set_status(StatusCode.OK)
            if not auto_close_span:
                span.end()
        else:
            ex:Exception = None
            to_wrap = get_wrapper_with_next_processor(to_wrap, handler, span, parent_span, instance, args, kwargs)
            if has_more_processors(to_wrap):
                try:
                    handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, None, span, parent_span, ex,
                                is_post_exec=False)
                except Exception as e:
                    logger.info(f"Warning: Error occurred in hydrate_span pre_process_span: {e}")
                try:
                    with monocle_trace_scope(get_builtin_scope_names(to_wrap)):
                        async for item in amonocle_iter_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, False, args, kwargs):
                            last_item = item
                            yield item
                except Exception as e:
                    ex = e
                    raise
                finally:
                    last_item = post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, last_item, span, parent_span, ex)
            else:
                try:
                    handler.hydrate_span(to_wrap, wrapped, instance, args, kwargs, None, span, parent_span, ex,
                                is_post_exec=False)
                except Exception as e:
                    logger.info(f"Warning: Error occurred in hydrate_span pre_process_span: {e}")
                try:
                    skip_execution, last_item = SpanHandler.skip_execution(span)
                    if not skip_execution:
                        with SpanHandler.workflow_type(to_wrap, span):
                            async for item in wrapped(*args, **kwargs):
                                last_item = item
                                yield item
                    else:
                        yield last_item
                except Exception as e:
                    ex = e
                    raise
                finally:
                    def post_process_span_internal(ret_val):
                        ret_val = post_process_span(handler, to_wrap, wrapped, instance, args, kwargs, ret_val, span, parent_span, ex)
                        if not auto_close_span:
                            span.end()
                    if ex is None and not auto_close_span and to_wrap.get("output_processor") and to_wrap.get("output_processor").get("response_processor"):
                        to_wrap.get("output_processor").get("response_processor")(to_wrap, None, post_process_span_internal)
                    else:
                        last_item = post_process_span_internal(last_item)
    return

async def amonocle_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return_value = None
    token = None
    pre_trace_token = None
    try:
        try:
            pre_trace_token = handler.pre_tracing(to_wrap, wrapped, instance, args, kwargs)
        except Exception as e:
            logger.info(f"Warning: Error occurred in pre_tracing: {e}")
        if to_wrap.get('skip_span', False) or handler.skip_span(to_wrap, wrapped, instance, args, kwargs):
            return_value = await wrapped(*args, **kwargs)
        else:
            add_workflow_span = get_value(ADD_NEW_WORKFLOW) == True
            token = attach(set_value(ADD_NEW_WORKFLOW, False))
            try:
                with monocle_trace_scope(get_builtin_scope_names(to_wrap)):
                    return_value, span_status = await amonocle_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, 
                                                                        add_workflow_span, args, kwargs)
            finally:
                detach(token)
        return return_value
    finally:
        try:
            handler.post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value, pre_trace_token)
        except Exception as e:
            logger.info(f"Warning: Error occurred in post_tracing: {e}")

async def amonocle_iter_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs) -> AsyncGenerator[any, None]:
    token = None
    pre_trace_token = None
    try:
        try:
            pre_trace_token = handler.pre_tracing(to_wrap, wrapped, instance, args, kwargs)
        except Exception as e:
            logger.info(f"Warning: Error occurred in pre_tracing: {e}")
        if to_wrap.get('skip_span', False) or handler.skip_span(to_wrap, wrapped, instance, args, kwargs):
            async for item in wrapped(*args, **kwargs):
                yield item
        else:
            add_workflow_span = get_value(ADD_NEW_WORKFLOW) == True
            token = attach(set_value(ADD_NEW_WORKFLOW, False))
            try:
                with monocle_trace_scope(get_builtin_scope_names(to_wrap)):
                    async for item in amonocle_iter_wrapper_span_processor(tracer, handler, to_wrap, wrapped, instance, source_path, add_workflow_span, args, kwargs):
                        yield item
            finally:
                detach(token)
        return
    finally:
        try:
            handler.post_tracing(to_wrap, wrapped, instance, args, kwargs, None, pre_trace_token)
        except Exception as e:
            logger.info(f"Warning: Error occurred in post_tracing: {e}")

@with_tracer_wrapper
def task_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return monocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)

@with_tracer_wrapper
async def atask_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    return await amonocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)

@with_tracer_wrapper
async def atask_iter_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs) -> AsyncGenerator[any, None]:
    async for item in amonocle_iter_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs):
        yield item
    return

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
    scope_values = evaluate_scope_values(args, kwargs, to_wrap, scope_values)
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
    scope_values = evaluate_scope_values(args, kwargs, to_wrap, scope_values)
    token = None
    try:
        if scope_values:
            token = set_scopes(scope_values)
        return_value = await wrapped(*args, **kwargs)
        return return_value
    finally:
        if token:
            remove_scope(token)

def evaluate_scope_values(args, kwargs, to_wrap, scope_values):
    if callable(scope_values):
        try:
            scope_values = scope_values(args, kwargs)
        except Exception as e:
            logger.warning("Warning: Error occurred in evaluate_scope_values: %s", str(e))
            scope_values = None
    if isinstance(scope_values, dict):
        return scope_values
    return None

@contextmanager
def start_as_monocle_span(tracer: Tracer, name: str, auto_close_span: bool) -> Iterator["Span"]:
    """ Wrapper to OTEL start_as_current_span to isolate monocle and non monocle spans.
        This essentially links monocle and non-monocle spans separately which is default behavior.
        It can be optionally overridden by setting the environment variable MONOCLE_ISOLATE_SPANS to false.
    """
    if not ISOLATE_MONOCLE_SPANS:
        # If not isolating, use the default start_as_current_span
        yield tracer.start_as_current_span(name, end_on_exit=auto_close_span)
        return
    
    original_span = get_current_span()
    monocle_span_token = attach(set_span_in_context(get_current_monocle_span()))
    with tracer.start_as_current_span(name, end_on_exit=auto_close_span) as span:
        new_monocle_token = attach(set_monocle_span_in_context(span))
        original_span_token = attach(set_span_in_context(original_span))
        yield span
        detach(original_span_token)
        detach(new_monocle_token)
    detach(monocle_span_token)

def get_builtin_scope_names(to_wrap) -> str:
    output_processor = None
    if "output_processor" in to_wrap:
        output_processor = to_wrap.get("output_processor", None)
    if "output_processor_list" in to_wrap:
        for processor in to_wrap["output_processor_list"]:
            if processor.get("type", None) in AGENTIC_SPANS:
                output_processor = processor
                break

    span_type = output_processor.get("type", None) if output_processor and isinstance(output_processor, dict) else None
    if span_type and span_type in AGENTIC_SPANS:
        return span_type
    return None

def get_wrapper_with_next_processor(to_wrap, handler, instance, span, parent_span, args, kwargs):
    if has_more_processors(to_wrap):
        next_output_processor_list = to_wrap.get('output_processor_list',[]).copy()
        while len(next_output_processor_list) > 0:
            next_output_processor = next_output_processor_list.pop(0)
            if handler.should_skip(next_output_processor, instance, span, parent_span, args, kwargs):
                next_output_processor = None
            else:
                break
        to_wrap = to_wrap.copy()
        to_wrap['output_processor_list'] = next_output_processor_list
        to_wrap['output_processor'] = next_output_processor
    return to_wrap

def has_more_processors(to_wrap) -> bool:
    return len(to_wrap.get('output_processor_list', [])) > 0
