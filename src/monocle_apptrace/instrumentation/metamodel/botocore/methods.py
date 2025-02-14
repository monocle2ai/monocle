from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.botocore.entities.inference import (
    INFERENCE,
)

BOTOCORE_METHODS = [{

      "package": "botocore.client",
      "object": "ClientCreator",
      "method": "create_client",
      "wrapper_method": task_wrapper,
      "skip_span": True,
      "post_processor": {
            "module": "monocle_apptrace.instrumentation.metamodel.botocore._helper",
            "method": "botocore_processor"
    },
      "output_processor": INFERENCE

}
]