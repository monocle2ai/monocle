# Copyright (C) Okahu Inc 2023-2024. All rights reserved

import logging
from typing import Collection,List
from wrapt import wrap_function_wrapper
from opentelemetry.trace import get_tracer
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry import trace
from okahu_apptrace.wrap_common import CONTEXT_PROPERTIES_KEY
from okahu_apptrace.wrapper import INBUILT_METHODS_LIST, WrapperMethod
from okahu_apptrace.exporter import OkahuSpanExporter 
from opentelemetry.context import get_value, attach, set_value


logger = logging.getLogger(__name__)

_instruments = ("langchain >= 0.0.346",)

class OkahuInstrumentor(BaseInstrumentor):
    
    workflow_name: str = ""
    user_wrapper_methods: list[WrapperMethod] = []
    instrumented_method_list: list[object] = []
    
    def __init__(
            self,
            user_wrapper_methods: list[WrapperMethod] = []) -> None:
        self.user_wrapper_methods = user_wrapper_methods
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(instrumenting_module_name= __name__, tracer_provider= tracer_provider)

        user_method_list = [
            {
                "package": method.package,
                "object": method.object,
                "method": method.method,
                "span_name": method.span_name,
                "wrapper": method.wrapper,
            } for method in self.user_wrapper_methods]

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
           

def setup_okahu_telemetry(
        workflow_name: str,
        span_processors: List[SpanProcessor] = [],
        wrapper_methods: List[WrapperMethod] = []):
    resource = Resource(attributes={
        SERVICE_NAME: workflow_name
    })
    traceProvider = TracerProvider(resource=resource)
    okahuProcessor = BatchSpanProcessor(OkahuSpanExporter())
    okahuProcessor.on_start = on_processor_start
    for processor in span_processors:
        processor.on_start = on_processor_start
        traceProvider.add_span_processor(processor)
    traceProvider.add_span_processor(okahuProcessor)
    trace.set_tracer_provider(traceProvider)
    instrumentor = OkahuInstrumentor(user_wrapper_methods=wrapper_methods)
    instrumentor.app_name = workflow_name
    instrumentor.instrument()


def on_processor_start(span: Span, parent_context):
    context_properties = get_value(CONTEXT_PROPERTIES_KEY)
    if context_properties is not None:
        for key, value in context_properties.items():
            span.set_attribute(
                f"{CONTEXT_PROPERTIES_KEY}.{key}", value
            )    

def set_context_properties(properties: dict) -> None:
    attach(set_value(CONTEXT_PROPERTIES_KEY, properties))




