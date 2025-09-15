from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.botocore.entities.inference import (
    INFERENCE,
)

BOTOCORE_METHODS = [
    {
      "package": "botocore.client",
      "object": "ClientCreator",
      "method": "create_client",
      "wrapper_method": task_wrapper,
      "span_handler":"botocore_handler",
      "output_processor": INFERENCE,
      "skip_span": True
    }
]