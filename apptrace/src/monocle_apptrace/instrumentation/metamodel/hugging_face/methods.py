from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.hugging_face.entities.agent import (
    AGENT,
    AGENT_REQUEST,
    TOOL,
)
from monocle_apptrace.instrumentation.metamodel.hugging_face.entities.inference import (
    INFERENCE,
)

HUGGING_FACE_METHODS = [
    {
        "package": "huggingface_hub",
        "object": "InferenceClient",
        "method": "chat_completion",  # sync
        "wrapper_method": task_wrapper,
        "span_handler": "non_framework_handler",
        "output_processor": INFERENCE,
    },
    {
        "package": "huggingface_hub",
        "object": "AsyncInferenceClient",
        "method": "chat_completion",  # async
        "wrapper_method": atask_wrapper,
        "span_handler": "non_framework_handler",
        "output_processor": INFERENCE,
    },
    {
        "package": "smolagents.agents",
        "object": "MultiStepAgent",
        "method": "run",
        "wrapper_method": task_wrapper,
        "span_handler": "non_framework_handler",
        "output_processor": AGENT_REQUEST,
    },
    {
        "package": "smolagents.agents",
        "object": "CodeAgent",
        "method": "run",
        "wrapper_method": task_wrapper,
        "span_handler": "non_framework_handler",
        "output_processor": AGENT,
    },
    {
        "package": "smolagents.agents",
        "object": "ToolCallingAgent",
        "method": "run",
        "wrapper_method": task_wrapper,
        "span_handler": "non_framework_handler",
        "output_processor": AGENT,
    },
    {
        "package": "smolagents.tools",
        "object": "Tool",
        "method": "__call__",
        "wrapper_method": task_wrapper,
        "span_handler": "non_framework_handler",
        "output_processor": TOOL,
    },
]
