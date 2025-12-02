import logging
import inspect
from typing import Collection, Dict, List, Union
import uuid
import inspect
from opentelemetry import trace
from opentelemetry.context import attach, get_value, set_value
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper
from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler, NonFrameworkSpanHandler
from monocle_apptrace.instrumentation.common.wrapper_method import (
    DEFAULT_METHODS_LIST,
    WrapperMethod,
    MONOCLE_SPAN_HANDLERS
)
from monocle_apptrace.instrumentation.common.wrapper import scope_wrapper, ascope_wrapper, monocle_wrapper, amonocle_wrapper
from monocle_apptrace.instrumentation.common.utils import (
    load_scopes,
    setup_readablespan_patch
)
from monocle_apptrace.instrumentation.common.constants import MONOCLE_INSTRUMENTOR
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
        monocle_exporters_list:str = None) -> MonocleInstrumentor:
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
    
    # Monkey-patch ReadableSpan.to_json to remove 0x prefix from trace_id/span_id
    setup_readablespan_patch()
    
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

def is_valid_trace_id_uuid(traceId: str) -> bool:
    try:
        uuid.UUID(traceId)
        return True
    except:
        pass
    return False

from monocle_apptrace.instrumentation.common.method_wrappers import (
    monocle_trace,
    amonocle_trace,
    monocle_trace_method,
    monocle_trace_http_route,
    start_trace,
    stop_trace,
    http_route_handler
)

