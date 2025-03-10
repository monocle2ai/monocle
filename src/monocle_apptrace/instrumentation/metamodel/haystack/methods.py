from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.haystack.entities.inference import INFERENCE
from monocle_apptrace.instrumentation.metamodel.haystack.entities.retrieval import RETRIEVAL

HAYSTACK_METHODS = [
    {
        "package": "haystack.components.retrievers.in_memory",
        "object": "InMemoryEmbeddingRetriever",
        "method": "run",
        "span_name": "haystack.retriever",
        "wrapper_method": task_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "haystack_integrations.components.retrievers.opensearch",
        "object": "OpenSearchEmbeddingRetriever",
        "method": "run",
        "span_name": "haystack.retriever",
        "wrapper_method": task_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "haystack.components.generators.openai",
        "object": "OpenAIGenerator",
        "method": "run",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "haystack.components.generators.chat.openai",
        "object": "OpenAIChatGenerator",
        "method": "run",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "haystack.core.pipeline.pipeline",
        "object": "Pipeline",
        "method": "run",
        "wrapper_method": task_wrapper,
        "span_type": "workflow"
    }
]
