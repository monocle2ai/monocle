from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.openai.entities.inference import (
    INFERENCE,
)

OPENAI_METHODS = [
    {
        "package": "openai",
        "object": "chat.completions",
        "method": "create",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
]
