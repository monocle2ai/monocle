from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.langchain.inference import (
    INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.langchain.retrieval import (
    RETRIEVAL,
)

LANGCHAIN_METHODS = [
    {
        "package": "langchain.prompts.base",
        "object": "BasePromptTemplate",
        "method": "invoke",
        "framework_type": "langchain",
        "wrapper_method": task_wrapper
    },
    {
        "package": "langchain.prompts.base",
        "object": "BasePromptTemplate",
        "method": "ainvoke",
        "framework_type": "langchain",
        "wrapper_method": atask_wrapper
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "invoke",
        "framework_type": "langchain",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "ainvoke",
        "framework_type": "langchain",
        "wrapper_method": atask_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "LLM",
        "method": "_generate",
        "framework_type": "langchain",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "LLM",
        "method": "_agenerate",
        "framework_type": "langchain",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "BaseLLM",
        "method": "invoke",
        "framework_type": "langchain",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "BaseLLM",
        "method": "ainvoke",
        "framework_type": "langchain",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain_core.retrievers",
        "object": "BaseRetriever",
        "method": "invoke",
        "framework_type": "langchain",
        "wrapper_method": task_wrapper,
        "output_processor": RETRIEVAL

    },
    {
        "package": "langchain_core.retrievers",
        "object": "BaseRetriever",
        "method": "ainvoke",
        "framework_type": "langchain",
        "wrapper_method": atask_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "langchain.schema",
        "object": "BaseOutputParser",
        "method": "invoke",
        "framework_type": "langchain",
        "wrapper_method": task_wrapper
    },
    {
        "package": "langchain.schema",
        "object": "BaseOutputParser",
        "method": "ainvoke",
        "framework_type": "langchain",
        "wrapper_method": atask_wrapper
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "invoke",
        "framework_type": "langchain",
        "span_name": "langchain.workflow",
        "wrapper_method": task_wrapper
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "ainvoke",
        "framework_type": "langchain",
        "span_name": "langchain.workflow",
        "wrapper_method": atask_wrapper
    }
]
