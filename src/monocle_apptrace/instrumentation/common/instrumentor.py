import logging
from typing import Collection, Dict, List, Union
import random
import uuid
from opentelemetry import trace
from contextlib import contextmanager
from opentelemetry.context import attach, get_value, set_value, get_current, detach
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import SpanContext
from opentelemetry.sdk.trace import TracerProvider, Span, id_generator
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper
from opentelemetry.trace.propagation import set_span_in_context
from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.wrapper_method import (
    DEFAULT_METHODS_LIST,
    WrapperMethod,
    MONOCLE_SPAN_HANDLERS
)
from monocle_apptrace.instrumentation.common.wrapper import scope_wrapper
from monocle_apptrace.instrumentation.common.utils import (
    set_scope, remove_scope, http_route_handler, load_scopes
)
from monocle_apptrace.instrumentation.common.constants import MONOCLE_INSTRUMENTOR
from functools import wraps
logger = logging.getLogger(__name__)

SESSION_PROPERTIES_KEY = "session"

_instruments = ()

monocle_tracer_provider: TracerProvider = None

class MonocleInstrumentor(BaseInstrumentor):
    workflow_name: str = ""
    user_wrapper_methods: list[Union[dict,WrapperMethod]] = []
    instrumented_method_list: list[object] = []
    handlers:Dict[str,SpanHandler] = {} # dict of handlers
    union_with_default_methods: bool = False

    def __init__(
            self,
            handlers,
            user_wrapper_methods: list[Union[dict,WrapperMethod]] = None,
            union_with_default_methods: bool = True
            ) -> None:
        self.user_wrapper_methods = user_wrapper_methods or []
        self.handlers = handlers
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
            @wraps(fn)
            def with_instrumentation(*args, **kwargs):
                handler = SpanHandler()
                with tracer.start_as_current_span(span_name) as span:
                    response = fn(*args, **kwargs)
                    handler.hydrate_span(to_wrap, span=span, wrapped=wrapped, instance=instance, args=args, kwargs=kwargs,
                                         result=response)
                    return response

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
        union_with_default_methods: bool = True) -> None:
    resource = Resource(attributes={
        SERVICE_NAME: workflow_name
    })
    exporters = get_monocle_exporter()
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
    instrumentor = MonocleInstrumentor(user_wrapper_methods=wrapper_methods or [], 
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


def propagate_trace_id(traceId = "", use_trace_context = False):
    try:
        if traceId.startswith("0x"):
            traceId = traceId.lstrip("0x")
        tracer = get_tracer(instrumenting_module_name= MONOCLE_INSTRUMENTOR, tracer_provider= get_tracer_provider())
        initial_id_generator = tracer.id_generator
        _parent_span_context = get_current() if use_trace_context else None
        if traceId and is_valid_trace_id_uuid(traceId):
            tracer.id_generator = FixedIdGenerator(uuid.UUID(traceId).int)

        span = tracer.start_span(name = "parent_placeholder_span", context= _parent_span_context)
        updated_span_context = set_span_in_context(span=span, context= _parent_span_context)
        updated_span_context = set_value("root_span_id", span.get_span_context().span_id, updated_span_context)
        token = attach(updated_span_context)

        span.end()
        tracer.id_generator = initial_id_generator
        return token
    except:
        logger.warning("Failed to propagate trace id")
    return


def propagate_trace_id_from_traceparent():
    propagate_trace_id(use_trace_context = True)


def stop_propagate_trace_id(token) -> None:
    try:
        detach(token)
    except:
        logger.warning("Failed to stop propagating trace id")


def is_valid_trace_id_uuid(traceId: str) -> bool:
    try:
        uuid.UUID(traceId)
        return True
    except:
        pass
    return False

def start_scope(scope_name: str, scope_value:str = None) -> object:
    return set_scope(scope_name, scope_value)

def stop_scope(token:object) -> None:
    remove_scope(token)
    return

@contextmanager
def monocle_trace_scope(scope_name: str, scope_value:str = None):
    token = start_scope(scope_name, scope_value)
    try:
        yield
    finally:
        stop_scope(token)

def monocle_trace_scope_method(scope_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            token = start_scope(scope_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                stop_scope(token)
        return wrapper
    return decorator

def monocle_trace_http_route(func):
    def wrapper(req):
        return http_route_handler(req.headers, func, req)
    return wrapper

class FixedIdGenerator(id_generator.IdGenerator):
    def __init__(
            self,
            trace_id: int) -> None:
        self.trace_id = trace_id

    def generate_span_id(self) -> int:
        return random.getrandbits(64)

    def generate_trace_id(self) -> int:
        return self.trace_id
