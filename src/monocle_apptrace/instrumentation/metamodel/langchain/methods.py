from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.langchain.entities.inference import (
    INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.langchain.entities.retrieval import (
    RETRIEVAL,
)

LANGCHAIN_METHODS = [
    {
        "package": "langchain.prompts.base",
        "object": "BasePromptTemplate",
        "method": "invoke",
        "wrapper_method": task_wrapper,
        "span_type": "workflow"
    },
    {
        "package": "langchain.prompts.base",
        "object": "BasePromptTemplate",
        "method": "ainvoke",
        "wrapper_method": atask_wrapper,
        "span_type": "workflow"
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "invoke",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "ainvoke",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "LLM",
        "method": "_generate",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "LLM",
        "method": "_agenerate",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "BaseLLM",
        "method": "invoke",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "BaseLLM",
        "method": "ainvoke",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "langchain_core.retrievers",
        "object": "BaseRetriever",
        "method": "invoke",
        "wrapper_method": task_wrapper,
        "output_processor": RETRIEVAL

    },
    {
        "package": "langchain_core.retrievers",
        "object": "BaseRetriever",
        "method": "ainvoke",
        "wrapper_method": atask_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "langchain.schema",
        "object": "BaseOutputParser",
        "method": "invoke",
        "wrapper_method": task_wrapper,
        "span_type": "workflow"
    },
    {
        "package": "langchain.schema",
        "object": "BaseOutputParser",
        "method": "ainvoke",
        "wrapper_method": atask_wrapper,
        "span_type": "workflow"
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "invoke",
        "span_name": "langchain.workflow",
        "wrapper_method": task_wrapper,
        "span_type": "workflow"
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "ainvoke",
        "span_name": "langchain.workflow",
        "wrapper_method": atask_wrapper,
        "span_type": "workflow"
    }
]
