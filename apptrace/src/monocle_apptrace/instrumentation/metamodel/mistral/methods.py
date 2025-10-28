from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper
from monocle_apptrace.instrumentation.metamodel.mistral.entities.inference import MISTRAL_INFERENCE
from monocle_apptrace.instrumentation.metamodel.mistral.entities.retrieval import MISTRAL_RETRIEVAL

MISTRAL_METHODS = [
    {
        "package": "mistralai.chat",          # where Chat is defined
        "object": "Chat",                     # class name
        "method": "complete",                 # the sync method
        "span_handler": "non_framework_handler",
        "wrapper_method": task_wrapper,
        "output_processor": MISTRAL_INFERENCE
    },
    {
        "package": "mistralai.chat",          # where Chat is defined
        "object": "Chat",                     # class name
        "method": "complete_async",           # the async method
        "span_handler": "non_framework_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": MISTRAL_INFERENCE
    },
    {
        "package": "mistralai.chat",
        "object": "Chat",
        "method": "stream",              # sync streaming
        "span_handler": "non_framework_handler",
        "wrapper_method": task_wrapper,
        "output_processor": MISTRAL_INFERENCE,
    },
    {
        "package": "mistralai.chat",
        "object": "Chat",
        "method": "stream_async",        # async streaming
        "span_handler": "non_framework_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": MISTRAL_INFERENCE,
    },
    {
        "package": "mistralai.embeddings",    # where Embeddings is defined
        "object": "Embeddings",               # sync embeddings client
        "method": "create",                   # sync create
        "span_handler": "non_framework_handler",
        "wrapper_method": task_wrapper,
        "output_processor": MISTRAL_RETRIEVAL
    },
]

 


