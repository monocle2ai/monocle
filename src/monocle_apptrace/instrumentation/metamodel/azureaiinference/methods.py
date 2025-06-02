from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.azureaiinference.entities.inference import INFERENCE

AZURE_AI_INFERENCE_METHODS = [
    # Chat Completions - Synchronous
    {
        "package": "azure.ai.inference",
        "object": "ChatCompletionsClient",
        "method": "complete",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    # Chat Completions - Asynchronous
    {
        "package": "azure.ai.inference.aio",
        "object": "ChatCompletionsClient",
        "method": "complete",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    }
]
