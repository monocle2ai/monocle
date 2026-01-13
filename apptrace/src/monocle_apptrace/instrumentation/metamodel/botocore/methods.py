from monocle_apptrace.instrumentation.common.wrapper import task_wrapper

BOTOCORE_METHODS = [
    {
      "package": "botocore.client",
      "object": "ClientCreator",
      "method": "create_client",
      "wrapper_method": task_wrapper,
      "span_handler":"botocore_handler",
      "skip_span": True
    }
]