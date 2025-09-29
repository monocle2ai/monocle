from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.ollama.entities.inference import (
    INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.ollama.entities.retrieval import (
    RETRIEVAL,
)

OLLAMA_METHODS = [
    # Module-level functions (these are the ones typically used)
    {
        "package": "ollama",
        "object": "",
        "method": "chat", 
        "wrapper_method": task_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "ollama",
        "object": "",
        "method": "generate",
        "wrapper_method": task_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "ollama",
        "object": "",
        "method": "embed",
        "wrapper_method": task_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": RETRIEVAL
    },
    {
        "package": "ollama",
        "object": "",
        "method": "embeddings",
        "wrapper_method": task_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": RETRIEVAL
    },
    # Client class methods
    {
        "package": "ollama._client",
        "object": "Client",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "ollama._client",
        "object": "AsyncClient",
        "method": "chat",
        "wrapper_method": atask_wrapper,
        "span_handler": "ollama_handler", 
        "output_processor": INFERENCE
    },
    {
        "package": "ollama._client",
        "object": "Client",
        "method": "generate",
        "wrapper_method": task_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "ollama._client",
        "object": "AsyncClient",
        "method": "generate",
        "wrapper_method": atask_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "ollama._client",
        "object": "Client",
        "method": "embed",
        "wrapper_method": task_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": RETRIEVAL
    },
    {
        "package": "ollama._client",
        "object": "AsyncClient",
        "method": "embed",
        "wrapper_method": atask_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": RETRIEVAL
    },
    {
        "package": "ollama._client",
        "object": "Client",
        "method": "embeddings",
        "wrapper_method": task_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": RETRIEVAL
    },
    {
        "package": "ollama._client",
        "object": "AsyncClient",
        "method": "embeddings",
        "wrapper_method": atask_wrapper,
        "span_handler": "ollama_handler",
        "output_processor": RETRIEVAL
    }
]
