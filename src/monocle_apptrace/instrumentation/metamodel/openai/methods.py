from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.openai.entities.inference import (
    INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.openai.entities.retrieval import (
    RETRIEVAL,
)

OPENAI_METHODS = [
    {
        "package": "openai.resources.chat.completions",
        "object": "Completions",
        "method": "create",
        "wrapper_method": task_wrapper,
        "span_handler": "non_framework_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "openai.resources.chat.completions",
        "object": "AsyncCompletions",
        "method": "create",
        "wrapper_method": atask_wrapper,
        "span_handler": "non_framework_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "openai.resources.embeddings",
        "object": "Embeddings",
        "method": "create",
        "wrapper_method": task_wrapper,
        "span_name": "openai_embeddings",
        "span_handler": "non_framework_handler",
        "output_processor": RETRIEVAL
    },
    {
        "package": "openai.resources.embeddings",
        "object": "AsyncEmbeddings",
        "method": "create",
        "wrapper_method": atask_wrapper,
        "span_name": "openai_embeddings",
        "span_handler": "non_framework_handler",
        "output_processor": RETRIEVAL
    }

]
