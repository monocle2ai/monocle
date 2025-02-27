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
                instrumentor = self.get_instrumentor
                if instrumentor:
                    instrumented_method = instrumentor(to_wrap, wrapped, span_name, return_value, original_method)
                    setattr(return_value, method_name, instrumented_method)

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, span):
        self._botocore_processor(to_wrap=to_wrap, wrapped=wrapped, instance=instance, return_value=result, args=args,
                                 kwargs=kwargs)
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, span)