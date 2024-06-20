

from monocle_apptrace.wrap_common import allm_wrapper, atask_wrapper, llm_wrapper, task_wrapper

def get_llm_span_name_for_openai(instance):
    if (hasattr(instance, "_is_azure_client") 
        and callable(getattr(instance, "_is_azure_client"))
        and instance._is_azure_client()):
        return "llamaindex.azure_openai"
    return "llamaindex.openai"

LLAMAINDEX_METHODS = [
    {
        "package": "llama_index.core.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "retrieve",
        "span_name": "llamaindex.retrieve",
        "wrapper": task_wrapper
    },
    {
        "package": "llama_index.core.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "aretrieve",
        "span_name": "llamaindex.retrieve",
        "wrapper": atask_wrapper
    },
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "query",
        "span_name": "llamaindex.query",
        "wrapper": task_wrapper,
    },
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "aquery",
        "span_name": "llamaindex.query",
        "wrapper": atask_wrapper,
    },
    {
        "package": "llama_index.core.llms.custom",
        "object": "CustomLLM",
        "method": "chat",
        "span_name": "llamaindex.llmchat",
        "wrapper": task_wrapper,
    },
    {
        "package": "llama_index.core.llms.custom",
        "object": "CustomLLM",
        "method": "achat",
        "span_name": "llamaindex.llmchat",
        "wrapper": atask_wrapper,
    },
    {
        "package": "llama_index.llms.openai.base",
        "object": "OpenAI",
        "method": "chat",
        "span_name": "llamaindex.openai",
        "span_name_getter" : get_llm_span_name_for_openai,
        "wrapper": llm_wrapper,
    },
     {
        "package": "llama_index.llms.openai.base",
        "object": "OpenAI",
        "method": "achat",
        "span_name": "llamaindex.openai",
        "wrapper": allm_wrapper,
    }
]

