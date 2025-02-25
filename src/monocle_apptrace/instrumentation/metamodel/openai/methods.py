from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.openai.entities.inference import (
    INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.openai.entities.retrieval import (
    RETRIEVAL,
)

OPENAI_METHODS = [
    {
        "package": "openai",
        "object": "chat.completions",
        "method": "create",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "openai",
        "object": "embeddings",
        "method": "create",
        "wrapper_method": task_wrapper,
        "output_processor": RETRIEVAL
    },
]
