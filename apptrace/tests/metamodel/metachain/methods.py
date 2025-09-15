from metamodel.metachain.entities.inference import (
    INFERENCE,
)

from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper

METACHAIN_METHODS = [

    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "invoke",
        "wrapper_method": task_wrapper,
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
    }
]