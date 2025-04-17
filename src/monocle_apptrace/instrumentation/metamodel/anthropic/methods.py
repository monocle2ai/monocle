from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.anthropic.entities.inference import (
    INFERENCE,
)

ANTHROPIC_METHODS = [
    {
        "package": "anthropic.resources",
        "object": "Messages",
        "method": "create",
        "wrapper_method": task_wrapper,
        "span_handler": "non_framework_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "anthropic.resources",
        "object": "AsyncMessages",
        "method": "create",
        "wrapper_method": atask_wrapper,
        "span_handler": "non_framework_handler",
        "output_processor": INFERENCE
    },

]
