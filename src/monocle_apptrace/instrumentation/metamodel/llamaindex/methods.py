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
        "span_name": "llamaindex.retrieve",
        "wrapper_method": task_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "llama_index.core.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "aretrieve",
        "span_name": "llamaindex.retrieve",
        "wrapper_method": atask_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "query",
        "span_name": "llamaindex.query",
        "wrapper_method": task_wrapper,
        "span_type": "workflow"
    },
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "aquery",
        "span_name": "llamaindex.query",
        "wrapper_method": atask_wrapper,
        "span_type": "workflow"
    },
    {
        "package": "llama_index.core.llms.custom",
        "object": "CustomLLM",
        "method": "chat",
        "span_name": "llamaindex.llmchat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.core.llms.custom",
        "object": "CustomLLM",
        "method": "achat",
        "span_name": "llamaindex.llmchat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE,
        
    },
    {
        "package": "llama_index.llms.openai.base",
        "object": "OpenAI",
        "method": "chat",
        "span_name": "llamaindex.openai",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.openai.base",
        "object": "OpenAI",
        "method": "achat",
        "span_name": "llamaindex.openai",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.mistralai.base",
        "object": "MistralAI",
        "method": "chat",
        "span_name": "llamaindex.mistralai",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.mistralai.base",
        "object": "MistralAI",
        "method": "achat",
        "span_name": "llamaindex.mistralai",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.core.agent",
        "object": "ReActAgent",
        "method": "chat",
        "span_name": "react.agent",
        "wrapper_method": task_wrapper,
        "output_processor": AGENT
    }
]
