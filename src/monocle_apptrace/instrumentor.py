import logging, os
from typing import Collection, List
from wrapt import wrap_function_wrapper
from opentelemetry.trace import get_tracer
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry import trace
from opentelemetry.context import get_value, attach, set_value
from monocle_apptrace.utils import process_wrapper_method_config
from monocle_apptrace.wrap_common import SESSION_PROPERTIES_KEY
from monocle_apptrace.wrapper import INBUILT_METHODS_LIST, WrapperMethod
from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter

logger = logging.getLogger(__name__)

_instruments = ()

class MonocleInstrumentor(BaseInstrumentor):
    workflow_name: str = ""
    user_wrapper_methods: list[WrapperMethod] = []
    instrumented_method_list: list[object] = []

    def __init__(
            self,
            user_wrapper_methods: list[WrapperMethod] = None) -> None:
        self.user_wrapper_methods = user_wrapper_methods or []
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(instrumenting_module_name=__name__, tracer_provider=tracer_provider)

        user_method_list = [
            {
                "package": method.package,
                "object": method.object,
                "method": method.method,
                "span_name": method.span_name,
                "wrapper": method.wrapper,
                "output_processor": method.output_processor
            } for method in self.user_wrapper_methods]
        process_wrapper_method_config(user_method_list)
        final_method_list = user_method_list + INBUILT_METHODS_LIST

        for wrapped_method in final_method_list:
            try:
                wrap_package = wrapped_method.get("package")
                wrap_object = wrapped_method.get("object")
                wrap_method = wrapped_method.get("method")
                wrapper = wrapped_method.get("wrapper")
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}" if wrap_object else wrap_method,
                    wrapper(tracer, wrapped_method),
                )
                self.instrumented_method_list.append(wrapped_method)
            except Exception as ex:
                if wrapped_method in user_method_list:
                    logger.error(f"""_instrument wrap Exception: {str(ex)}
                                for package: {wrap_package},
                                object:{wrap_object},
                                method:{wrap_method}""")

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
        wrapper_methods: List[WrapperMethod] = None):
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
    instrumentor = MonocleInstrumentor(user_wrapper_methods=wrapper_methods or [])
    # instrumentor.app_name = workflow_name
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()

def on_processor_start(span: Span, parent_context):
    context_properties = get_value(SESSION_PROPERTIES_KEY)
    if context_properties is not None:
        for key, value in context_properties.items():
            span.set_attribute(
                f"{SESSION_PROPERTIES_KEY}.{key}", value
            )

def set_context_properties(properties: dict) -> None:
    attach(set_value(SESSION_PROPERTIES_KEY, properties))