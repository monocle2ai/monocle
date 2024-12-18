from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.botocore.inference import (
    INFERENCE,
)

BOTOCORE_METHODS = [{

      "package": "botocore.client",
      "object": "ClientCreator",
      "method": "create_client",
      "wrapper_package": "wrap_common",
      "wrapper_method": task_wrapper,
      "skip_span": True,
      "output_processor": INFERENCE

}
]