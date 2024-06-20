

from monocle_apptrace.wrap_common import allm_wrapper, atask_wrapper, llm_wrapper, task_wrapper

LANGCHAIN_METHODS = [
    {
        "package": "langchain.prompts.base",
        "object": "BasePromptTemplate",
        "method": "invoke",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.prompts.base",
        "object": "BasePromptTemplate",
        "method": "ainvoke",
        "wrapper": atask_wrapper,
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "invoke",
        "wrapper": llm_wrapper,
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "ainvoke",
        "wrapper": allm_wrapper,
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "LLM",
        "method": "_generate",
        "wrapper": llm_wrapper,
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "LLM",
        "method": "_agenerate",
        "wrapper": llm_wrapper,
    },
    {
        "package": "langchain_core.retrievers",
        "object": "BaseRetriever",
        "method": "invoke",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain_core.retrievers",
        "object": "BaseRetriever",
        "method": "ainvoke",
        "wrapper": atask_wrapper,
    },
    {
        "package": "langchain.schema",
        "object": "BaseOutputParser",
        "method": "invoke",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.schema",
        "object": "BaseOutputParser",
        "method": "ainvoke",
        "wrapper": atask_wrapper,
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "invoke",
        "span_name": "langchain.workflow",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "ainvoke",
        "span_name": "langchain.workflow",
        "wrapper": atask_wrapper,
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableParallel",
        "method": "invoke",
        "span_name": "langchain.workflow",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableParallel",
        "method": "ainvoke",
        "span_name": "langchain.workflow",
        "wrapper": atask_wrapper,
    },
    
]
