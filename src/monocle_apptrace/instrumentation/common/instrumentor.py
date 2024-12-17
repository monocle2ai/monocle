import logging
from typing import Collection, Dict, List, Union

from opentelemetry import trace
from opentelemetry.context import attach, get_value, set_value
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.wrapper_method import (
    DEFAULT_METHODS_LIST,
    WrapperMethod,
)

logger = logging.getLogger(__name__)

SESSION_PROPERTIES_KEY = "session"

_instruments = ()

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
        self.handlers = handlers or {'default':SpanHandler()}
        self.union_with_default_methods = union_with_default_methods
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider: TracerProvider = kwargs.get("tracer_provider")
        tracer = get_tracer(instrumenting_module_name="monocle_apptrace", tracer_provider=tracer_provider)

        final_method_list = []
        if self.union_with_default_methods is True:
            final_method_list= final_method_list + DEFAULT_METHODS_LIST

        for method in self.user_wrapper_methods:
            if isinstance(method, dict):
                final_method_list.append(method)
            elif isinstance(method, WrapperMethod):
                final_method_list.append(method.to_dict())
        
        for method_config in final_method_list:
            target_package = method_config.get("package", None)
            target_object = method_config.get("object", None)
            target_method = method_config.get("method", None)
            wrapped_by = method_config.get("wrapper_method", None)
            #get the requisite handler or default one
            handler_key = method_config.get("span_handler",'default')
            try:
                handler =  self.handlers.get(handler_key)
                wrap_function_wrapper(
                    target_package,
                    f"{target_object}.{target_method}" if target_object else target_method,
                    wrapped_by(tracer, handler, method_config),
                )
                self.instrumented_method_list.append(method_config)
            except Exception as ex:
                logger.error(f"""_instrument wrap Exception: {str(ex)}
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
                logger.error(f"""_instrument unwrap Exception: {str(ex)}
                             for package: {wrap_package},
                             object:{wrap_object},
                             method:{wrap_method}""")

def setup_monocle_telemetry(
        workflow_name: str,
        span_processors: List[SpanProcessor] = None,
        span_handlers: Dict[str,SpanHandler] = None,
        wrapper_methods: List[Union[dict,WrapperMethod]] = None,
        union_with_default_methods: bool = True) -> None:
    resource = Resource(attributes={
        SERVICE_NAME: workflow_name
    })
    span_processors = span_processors or [BatchSpanProcessor(get_monocle_exporter())]
    trace_provider = TracerProvider(resource=resource)
    attach(set_value("workflow_name", workflow_name))
    tracer_provider_default = trace.get_tracer_provider()
    provider_type = type(tracer_provider_default).__name__
    is_proxy_provider = "Proxy" in provider_type
    for processor in span_processors:
        processor.on_start = on_processor_start
        if not is_proxy_provider:
            tracer_provider_default.add_span_processor(processor)
        else:
            trace_provider.add_span_processor(processor)
    if is_proxy_provider:
        trace.set_tracer_provider(trace_provider)
    instrumentor = MonocleInstrumentor(user_wrapper_methods=wrapper_methods or [], 
                                       handlers=span_handlers, union_with_default_methods = union_with_default_methods)
    # instrumentor.app_name = workflow_name
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(trace_provider=trace_provider)

def on_processor_start(span: Span, parent_context):
    context_properties = get_value(SESSION_PROPERTIES_KEY)
    if context_properties is not None:
        for key, value in context_properties.items():
            span.set_attribute(
                f"{SESSION_PROPERTIES_KEY}.{key}", value
            )

def set_context_properties(properties: dict) -> None:
    attach(set_value(SESSION_PROPERTIES_KEY, properties))