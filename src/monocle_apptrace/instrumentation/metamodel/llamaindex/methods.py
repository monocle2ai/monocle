from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.llamaindex.entities.inference import (
    INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.llamaindex.entities.agent import AGENT
from monocle_apptrace.instrumentation.metamodel.llamaindex.entities.retrieval import (
    RETRIEVAL,
)


LLAMAINDEX_METHODS = [
    {
        "package": "llama_index.core.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "retrieve",
        "wrapper_method": task_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "llama_index.core.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "aretrieve",
        "wrapper_method": atask_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "query",
        "wrapper_method": task_wrapper
    },
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "aquery",
        "wrapper_method": atask_wrapper
    },
    {
        "package": "llama_index.core.llms.custom",
        "object": "CustomLLM",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.core.llms.custom",
        "object": "CustomLLM",
        "method": "achat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE,
        
    },
    {
        "package": "llama_index.llms.openai.base",
        "object": "OpenAI",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.openai.base",
        "object": "OpenAI",
        "method": "achat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
        {
        "package": "llama_index.llms.openai.base",
        "object": "OpenAI",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.openai.base",
        "object": "OpenAI",
        "method": "achat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.mistralai.base",
        "object": "MistralAI",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.mistralai.base",
        "object": "MistralAI",
        "method": "achat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.core.agent",
        "object": "ReActAgent",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": AGENT
    },
    {
        "package": "llama_index.llms.anthropic",
        "object": "Anthropic",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.anthropic",
        "object": "Anthropic",
        "method": "achat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    }
]
