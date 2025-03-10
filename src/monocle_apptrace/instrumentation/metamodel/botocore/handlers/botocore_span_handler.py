from opentelemetry.context import get_value, set_value, attach, detach
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler

class BotoCoreSpanHandler(SpanHandler):

    def _botocore_processor(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        service_name = kwargs.get("service_name")
        service_method_mapping = {
            "sagemaker-runtime": "invoke_endpoint",
            "bedrock-runtime": "converse",
        }
        if service_name in service_method_mapping:
            method_name = service_method_mapping[service_name]
            original_method = getattr(return_value, method_name, None)
            span_name = "botocore-" + service_name + "-invoke-endpoint"
            # wrap_util(original_method, span_name)
            if original_method:
                instrumentor = self.instrumentor
                if instrumentor:
                    instrumented_method = instrumentor(to_wrap, wrapped, span_name, return_value, original_method)
                    setattr(return_value, method_name, instrumented_method)

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        self._botocore_processor(to_wrap=to_wrap, wrapped=wrapped, instance=instance, return_value=return_value, args=args,
                                 kwargs=kwargs)
        return super().pre_tracing(to_wrap, wrapped, instance, args, kwargs)
