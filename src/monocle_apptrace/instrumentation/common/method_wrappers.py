import logging
import inspect
from typing import Dict, List, Optional, Any
from functools import wraps
import inspect
from opentelemetry.context import attach, get_current, detach
from opentelemetry.sdk.trace import Span
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import get_tracer
from opentelemetry.trace.propagation import set_span_in_context, _SPAN_KEY
from contextlib import contextmanager, asynccontextmanager
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.common.utils import (
    set_scope, remove_scope, http_route_handler, http_async_route_handler
)
from monocle_apptrace.instrumentation.common.constants import MONOCLE_INSTRUMENTOR
from monocle_apptrace.instrumentation.common.instrumentor import get_tracer_provider

logger = logging.getLogger(__name__)




def start_trace(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None
):
    """
    Starts a new trace. All the spans created after this call will be part of the same trace. 
    
    Args:
        span_name: Optional custom span name. If None, uses the default span name.
        attributes: Optional dictionary of custom attributes to set on the span.
        events: Optional list of events to add to the span. Each event should be a dict with 
                'name' and optionally 'attributes' keys.
    
    Returns:
        Token: A token representing the attached context for the span.
                      This token is to be used later to stop the current trace.
                      Returns None if tracing fails.
    
    Raises:
        Exception: The function catches all exceptions internally and logs a warning.
    """
    try:
        tracer = get_tracer(instrumenting_module_name= MONOCLE_INSTRUMENTOR, tracer_provider= get_tracer_provider())
        span_name = span_name or "custom_span"
        span = tracer.start_span(name=span_name)
        updated_span_context = set_span_in_context(span=span)
        
        # Set default monocle attributes
        SpanHandler.set_default_monocle_attributes(span)
        if SpanHandler.is_root_span(span):
            SpanHandler.set_workflow_properties(span)
        
        # Set custom attributes and events using common method
        _setup_span_attributes_and_events(span, attributes, events)
        
        token = attach(updated_span_context)
        return token
    except Exception as e:
        logger.warning(f"Failed to start trace: {e}")
        return None

def stop_trace(
    token,
    final_attributes: Optional[Dict[str, Any]] = None,
    final_events: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Stop the active trace. All the spans created after this will not be part of the trace.
    
    Args:
        token: The token that was returned when the trace was started. Can be None in which case only the span is ended.
        final_attributes: Optional dictionary of final attributes to set on the span before ending.
        final_events: Optional list of final events to add to the span before ending.
    
    Returns:
        None
    """
    try:
        _parent_span_context = get_current()
        if _parent_span_context is not None:
            parent_span: Span = _parent_span_context.get(_SPAN_KEY, None)
            if parent_span is not None:
                # Set final attributes and events using common method
                _setup_span_attributes_and_events(parent_span, final_attributes, final_events)
                
                parent_span.end()
        if token is not None:
            detach(token)
    except Exception as e:
        logger.warning(f"Failed to stop trace: {e}")

def start_scope(
    scope_name: str, 
    scope_value: Optional[str] = None
) -> object:
    """
    Start a new scope with the given name and optional value. If no value is provided, a random UUID will be generated.
    All the spans, across traces created after this call will have the scope attached until the scope is stopped.
    
    Args:
        scope_name: The name of the scope.
        scope_value: Optional value of the scope. If None, a random UUID will be generated.
    
    Returns:
        Token: A token representing the attached context for the scope. This token is to be used later to stop the current scope.
    """
    try:
        # Set the scope using existing utility
        token = set_scope(scope_name, scope_value)
        return token
    except Exception as e:
        logger.warning(f"Failed to start scope: {e}")
        return None

def stop_scope(
    token: object
) -> None:
    """
    Stop the active scope. All the spans created after this will not have the scope attached.
    
    Args:
        token: The token that was returned when the scope was started.
    
    Returns:
        None
    """
    try:
        # Remove the scope
        remove_scope(token)
    except Exception as e:
        logger.warning(f"Failed to stop scope: {e}")
    return


@contextmanager
def monocle_trace(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None
):
    """
    Context manager to start and stop a trace. All the spans, across traces created within the encapsulated code will have same trace ID
    
    Args:
        span_name: Optional custom span name.
        attributes: Optional dictionary of custom attributes to set on the span.
        events: Optional list of events to add to the span at start.
    """
    try:
        tracer = get_tracer(instrumenting_module_name=MONOCLE_INSTRUMENTOR, tracer_provider=get_tracer_provider())
        span_name = span_name or "custom_span"
                
        with tracer.start_as_current_span(span_name) as span:
            # Set default monocle attributes
            SpanHandler.set_default_monocle_attributes(span)
            if SpanHandler.is_root_span(span):
                SpanHandler.set_workflow_properties(span)
            
            # Set custom attributes and events using common method
            _setup_span_attributes_and_events(span, attributes, events)
            
            try:
                yield
            finally:
                pass

            
    except Exception as e:
        logger.warning(f"Failed in monocle_trace: {e}")
        yield  # Still yield to not break the context manager

@asynccontextmanager
async def amonocle_trace(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None
):
    """
    Async context manager to start and stop a trace. All the spans, across traces created within the encapsulated code will have same trace ID
    
    Args:
        span_name: Optional custom span name.
        attributes: Optional dictionary of custom attributes to set on the span.
        events: Optional list of events to add to the span at start.
    """
    try:
        tracer = get_tracer(instrumenting_module_name=MONOCLE_INSTRUMENTOR, tracer_provider=get_tracer_provider())
        span_name = span_name or "custom_span"

        with tracer.start_as_current_span(span_name) as span:
            # Set default monocle attributes
            SpanHandler.set_default_monocle_attributes(span)
            if SpanHandler.is_root_span(span):
                SpanHandler.set_workflow_properties(span)
            
            # Set custom attributes and events using common method
            _setup_span_attributes_and_events(span, attributes, events)
            
            try:
                yield
            finally:
                pass
                
            
    except Exception as e:
        logger.warning(f"Failed in amonocle_trace: {e}")
        yield  # Still yield to not break the context manager

@contextmanager
def monocle_trace_scope(
    scope_name: str, 
    scope_value: Optional[str] = None
):
    """
    Context manager to start and stop a scope. All the spans, across traces created within the encapsulated code will have the scope attached.
    
    Args:
        scope_name: The name of the scope.
        scope_value: Optional value of the scope. If None, a random UUID will be generated.
    """
    token = start_scope(scope_name, scope_value)
    try:
        yield
    finally:
        stop_scope(token)

@asynccontextmanager
async def amonocle_trace_scope(
    scope_name: str, 
    scope_value: Optional[str] = None
):
    """
    Async context manager to start and stop a scope. All the spans, across traces created within the encapsulated code will have the scope attached.
    
    Args:
        scope_name: The name of the scope.
        scope_value: Optional value of the scope. If None, a random UUID will be generated.
    """
    token = start_scope(scope_name, scope_value)
    try:
        yield
    finally:
        stop_scope(token)
    
def monocle_trace_scope_method(
    scope_name: str, 
    scope_value: Optional[str] = None
):
    """
    Decorator to start and stop a scope for a method. All the spans, across traces created in the method will have the scope attached.
    
    Args:
        scope_name: The name of the scope.
        scope_value: Optional value of the scope. If None, a random UUID will be generated.
    """
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                async with amonocle_trace_scope(
                    scope_name, scope_value
                ):
                    result = await func(*args, **kwargs)
                    return result
            return wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with monocle_trace_scope(
                    scope_name, scope_value
                ):
                    result = func(*args, **kwargs)
                    return result
            return wrapper
    return decorator

def monocle_trace_method(
    span_name: Optional[str] = None
):
    """
    Decorator to start and stop a trace for a method. All the spans created in the method will be part of the same trace.
    
    Args:
        span_name: Optional custom span name. If None, uses the decorated function's name.
    """
    
    def decorator(func):
        tracer = get_tracer(instrumenting_module_name=MONOCLE_INSTRUMENTOR, tracer_provider=get_tracer_provider())
        handler = SpanHandler()
        source_path= func.__code__.co_filename + ":" + str(func.__code__.co_firstlineno)
        # Use function name as span name if not provided
        effective_span_name = span_name or func.__name__ or "custom_span"

        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await atask_wrapper(
                    tracer=tracer,
                    handler=handler,
                    to_wrap={
                        "span_name": effective_span_name,
                        "output_processor":{
                            "type": "custom",
                        }
                    }
                )(  wrapped=func,                        
                    instance=None,
                    source_path=source_path,
                    args=args,
                    kwargs=kwargs)
            return wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return task_wrapper(
                    tracer=tracer,
                    handler=handler,
                    to_wrap={
                        "span_name": effective_span_name,
                        "output_processor":{
                            "type": "custom",
                        }
                    }
                )(  wrapped=func,                        
                    instance=None,
                    source_path=source_path,
                    args=args,
                    kwargs=kwargs)
            return wrapper
    return decorator

def monocle_trace_http_route(func):
    """
    Decorator to start and stop a continue traces and scope for a http route. It will also initiate new scopes from the http headers if configured in ``monocle_scopes.json``
    All the spans, across traces created in the route will have the scope attached.
    """
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await http_async_route_handler(func, *args, **kwargs)
        return wrapper
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return http_route_handler(func, *args, **kwargs)
        return wrapper


def _setup_span_attributes_and_events(
    span,
    attributes: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Common method to set attributes and events on a span.
    
    Args:
        span: The span to configure
        attributes: Optional dictionary of custom attributes to set on the span
        events: Optional list of events to add to the span
    """
    # Set custom attributes if provided
    if attributes:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    
    # Add custom events if provided
    if events:
        for event in events:
            event_name = event.get('name')
            event_attributes = event.get('attributes', {})
            if event_name:
                span.add_event(event_name, event_attributes)

