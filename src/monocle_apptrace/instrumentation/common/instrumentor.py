import logging
import inspect
from typing import Collection, Dict, List, Union, Optional, Any
import random
import uuid
import inspect
from opentelemetry import trace
from contextlib import contextmanager, asynccontextmanager
from opentelemetry.context import attach, get_value, set_value, get_current, detach
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import SpanContext
from opentelemetry.sdk.trace import TracerProvider, Span, id_generator
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper
from opentelemetry.trace.propagation import set_span_in_context, _SPAN_KEY
from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler, NonFrameworkSpanHandler
from monocle_apptrace.instrumentation.common.wrapper_method import (
    DEFAULT_METHODS_LIST,
    WrapperMethod,
    MONOCLE_SPAN_HANDLERS
)
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, scope_wrapper, ascope_wrapper, monocle_wrapper, amonocle_wrapper, task_wrapper
from monocle_apptrace.instrumentation.common.utils import (
    set_scope, remove_scope, http_route_handler, load_scopes, http_async_route_handler
)
from monocle_apptrace.instrumentation.common.constants import MONOCLE_INSTRUMENTOR, WORKFLOW_TYPE_GENERIC
from functools import wraps
logger = logging.getLogger(__name__)

SESSION_PROPERTIES_KEY = "session"

_instruments = ()

monocle_tracer_provider: TracerProvider = None

class MonocleInstrumentor(BaseInstrumentor):
    workflow_name: str = ""
    user_wrapper_methods: list[Union[dict,WrapperMethod]] = [],
    exporters: list[SpanExporter] = [],
    instrumented_method_list: list[object] = []
    handlers:Dict[str,SpanHandler] = None # dict of handlers
    union_with_default_methods: bool = False

    def __init__(
            self,
            handlers,
            user_wrapper_methods: list[Union[dict,WrapperMethod]] = None,
            exporters: list[SpanExporter] = None,
            union_with_default_methods: bool = True
            ) -> None:
        self.user_wrapper_methods = user_wrapper_methods or []
        self.handlers = handlers
        self.exporters = exporters
        if self.handlers is not None:
            for key, val in MONOCLE_SPAN_HANDLERS.items():
                if key not in self.handlers:
                    self.handlers[key] = val
        else:
            self.handlers = MONOCLE_SPAN_HANDLERS
        self.union_with_default_methods = union_with_default_methods
        super().__init__()

    def get_instrumentor(self, tracer):
        def instrumented_endpoint_invoke(to_wrap,wrapped, span_name, instance,fn):
            if inspect.iscoroutinefunction(fn):
                @wraps(fn)
                async def with_instrumentation(*args, **kwargs):
                    boto_method_to_wrap = to_wrap.copy()
                    boto_method_to_wrap['skip_span'] = False
                    return await amonocle_wrapper(tracer, NonFrameworkSpanHandler(),
                            boto_method_to_wrap, fn, instance, "", args, kwargs)
            else:
                @wraps(fn)
                def with_instrumentation(*args, **kwargs):
                    boto_method_to_wrap = to_wrap.copy()
                    boto_method_to_wrap['skip_span'] = False
                    return monocle_wrapper(tracer, NonFrameworkSpanHandler(),
                            boto_method_to_wrap, fn, instance, "", args, kwargs)
            return with_instrumentation
        return instrumented_endpoint_invoke

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider: TracerProvider = kwargs.get("tracer_provider")
        set_tracer_provider(tracer_provider)
        tracer = get_tracer(instrumenting_module_name=MONOCLE_INSTRUMENTOR, tracer_provider=tracer_provider)

        final_method_list = []
        if self.union_with_default_methods is True:
            final_method_list= final_method_list + DEFAULT_METHODS_LIST

        for method in self.user_wrapper_methods:
            if isinstance(method, dict):
                final_method_list.append(method)
            elif isinstance(method, WrapperMethod):
                final_method_list.append(method.to_dict())

        for method in load_scopes():
            if method.get('async', False):
                method['wrapper_method'] = ascope_wrapper
            else:
                method['wrapper_method'] = scope_wrapper
            final_method_list.append(method)
        
        for method_config in final_method_list:
            target_package = method_config.get("package", None)
            target_object = method_config.get("object", None)
            target_method = method_config.get("method", None)
            wrapped_by = method_config.get("wrapper_method", None)
            #get the requisite handler or default one
            handler_key = method_config.get("span_handler",'default')
            try:
                handler =  self.handlers.get(handler_key)
                if not handler:
                    logger.warning("incorrect or empty handler falling back to default handler")
                    handler = self.handlers.get('default')
                handler.set_instrumentor(self.get_instrumentor(tracer))
                wrap_function_wrapper(
                    target_package,
                    f"{target_object}.{target_method}" if target_object else target_method,
                    wrapped_by(tracer, handler, method_config),
                )
                self.instrumented_method_list.append(method_config)
            except ModuleNotFoundError as e:
                logger.debug(f"ignoring module {e.name}")

            except Exception as ex:
                logger.error(f"""_instrument wrap exception: {str(ex)}
                            for package: {target_package},
                            object:{target_object},
                            method:{target_method}""")

    def _uninstrument(self, **kwargs):
        for wrapped_method in self.instrumented_method_list:
            try:
                wrap_package = wrapped_method.get("package")
                wrap_object = wrapped_method.get("object")
                wrap_method = wrapped_method.get("method")
                unwrap(
                    f"{wrap_package}.{wrap_object}" if wrap_object else wrap_package,
                    wrap_method,
                )
            except Exception as ex:
                logger.error(f"""_instrument unwrap exception: {str(ex)}
                             for package: {wrap_package},
                             object:{wrap_object},
                             method:{wrap_method}""")

def set_tracer_provider(tracer_provider: TracerProvider):
    global monocle_tracer_provider
    monocle_tracer_provider = tracer_provider

def get_tracer_provider() -> TracerProvider:
    global monocle_tracer_provider
    return monocle_tracer_provider

def setup_monocle_telemetry(
        workflow_name: str,
        span_processors: List[SpanProcessor] = None,
        span_handlers: Dict[str,SpanHandler] = None,
        wrapper_methods: List[Union[dict,WrapperMethod]] = None,
        union_with_default_methods: bool = True,
        monocle_exporters_list:str = None) -> None:
    """
    Set up Monocle telemetry for the application.

    Parameters
    ----------
    workflow_name : str
        The name of the workflow to be used as the service name in telemetry.
    span_processors : List[SpanProcessor], optional
        Custom span processors to use instead of the default ones. If None, 
        BatchSpanProcessors with Monocle exporters will be used. This can't be combined with `monocle_exporters_list`.
    span_handlers : Dict[str, SpanHandler], optional
        Dictionary of span handlers to be used by the instrumentor, mapping handler names to handler objects.
    wrapper_methods : List[Union[dict, WrapperMethod]], optional
        Custom wrapper methods for instrumentation. If None, default methods will be used.
    union_with_default_methods : bool, default=True
        If True, combine the provided wrapper_methods with the default methods.
        If False, only use the provided wrapper_methods.
    monocle_exporters_list : str, optional
        Comma-separated list of exporters to use. This will override the env setting MONOCLE_EXPORTERS.
        Supported exporters are: s3, blob, okahu, file, memory, console. This can't be combined with `span_processors`.
    """
    resource = Resource(attributes={
        SERVICE_NAME: workflow_name
    })
    if span_processors and monocle_exporters_list:
        raise ValueError("span_processors and monocle_exporters_list can't be used together")
    exporters:List[SpanExporter] = get_monocle_exporter(monocle_exporters_list)
    span_processors = span_processors or [BatchSpanProcessor(exporter) for exporter in exporters]
    set_tracer_provider(TracerProvider(resource=resource))
    attach(set_value("workflow_name", workflow_name))
    tracer_provider_default = trace.get_tracer_provider()
    provider_type = type(tracer_provider_default).__name__
    is_proxy_provider = "Proxy" in provider_type
    for processor in span_processors:
        processor.on_start = on_processor_start
        if not is_proxy_provider:
            tracer_provider_default.add_span_processor(processor)
        else:
            get_tracer_provider().add_span_processor(processor)
    if is_proxy_provider:
        trace.set_tracer_provider(get_tracer_provider())
    instrumentor = MonocleInstrumentor(user_wrapper_methods=wrapper_methods or [], exporters=exporters,
                                       handlers=span_handlers, union_with_default_methods = union_with_default_methods)
    # instrumentor.app_name = workflow_name
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(trace_provider=get_tracer_provider())

    return instrumentor

def on_processor_start(span: Span, parent_context):
    context_properties = get_value(SESSION_PROPERTIES_KEY)
    if context_properties is not None:
        for key, value in context_properties.items():
            span.set_attribute(
                f"{SESSION_PROPERTIES_KEY}.{key}", value
            )

def set_context_properties(properties: dict) -> None:
    attach(set_value(SESSION_PROPERTIES_KEY, properties))

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

def is_valid_trace_id_uuid(traceId: str) -> bool:
    try:
        uuid.UUID(traceId)
        return True
    except:
        pass
    return False

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


