from opentelemetry.context import get_value, set_value, attach, detach
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.metamodel.botocore.entities.inference import INFERENCE
from monocle_apptrace.instrumentation.metamodel.botocore.entities.retrieval import RETRIEVAL

class BotoCoreSpanHandler(SpanHandler):

    def _get_output_processor_for_method(self, method_name):
        """Determine which output processor to use based on the method name"""
        # Methods that generate retrieval spans
        retrieval_methods = ["retrieve_and_generate"]
        
        # Methods that generate inference spans (default for most)
        inference_methods = ["converse", "invoke_model", "invoke_data_automation_async", "invoke_endpoint"]
        
        if method_name in retrieval_methods:
            return RETRIEVAL
        elif method_name in inference_methods:
            return INFERENCE
        else:
            # Default to inference for unknown methods
            return INFERENCE

    def _botocore_processor(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        service_name = kwargs.get("service_name")
        service_method_mapping = {
            "sagemaker-runtime": ["invoke_endpoint"],
            "bedrock-runtime": ["converse", "invoke_model"],
            "bedrock-agent-runtime": ["retrieve_and_generate"],
            "bedrock-data-automation-runtime": ["invoke_data_automation_async"],
        }
        if service_name in service_method_mapping:
            method_names = service_method_mapping[service_name]
            for method_name in method_names:
                original_method = getattr(return_value, method_name, None)
                if original_method:
                    # e.g., "bedrock-runtime" -> "BedrockRuntime"
                    client_class_name = ''.join(word.capitalize() for word in service_name.split('-'))
                    span_name = f"botocore.client.{client_class_name}"
                    instrumentor = self.instrumentor
                    if instrumentor:
                        # Create a copy of to_wrap with the appropriate output_processor
                        to_wrap_copy = to_wrap.copy()
                        to_wrap_copy['output_processor'] = self._get_output_processor_for_method(method_name)
                        to_wrap_copy['span_name'] = span_name
                        instrumented_method = instrumentor(to_wrap_copy, wrapped, span_name, return_value, original_method)
                        setattr(return_value, method_name, instrumented_method)

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value,token=None):
        self._botocore_processor(to_wrap=to_wrap, wrapped=wrapped, instance=instance, return_value=return_value, args=args,
                                 kwargs=kwargs)
        return super().post_tracing(to_wrap, wrapped, instance, args, kwargs,return_value)
