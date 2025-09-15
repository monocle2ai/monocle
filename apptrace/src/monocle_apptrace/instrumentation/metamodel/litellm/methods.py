from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.litellm.entities.inference import INFERENCE

LITELLM_METHODS = [
    {
        "package": "litellm.llms.openai.openai",
        "object": "OpenAIChatCompletion",
        "method": "completion",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "litellm.llms.azure.azure",
        "object": "AzureChatCompletion",
        "method": "completion",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    }
]
