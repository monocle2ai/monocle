from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.xai.entities.inference import (
    INFERENCE,
)

XAI_METHODS = [
    #{
    #    "package": "xai_sdk.chat",
    #    "object": "BaseClient",
    #    "method": "create",
    #    "wrapper_method": task_wrapper,
    #    "span_handler": "xai_handler",
    #    "output_processor": INFERENCE
    #},
    {
        "package": "xai_sdk.sync.chat",
        "object": "Chat",
        "method": "sample",
        "wrapper_method": task_wrapper,
        "span_handler": "xai_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "xai_sdk.aio.chat",
        "object": "Chat",
        "method": "sample",
        "wrapper_method": atask_wrapper,
        "span_handler": "xai_handler",
        "output_processor": INFERENCE
    },
    #{
    #    "package": "xai_sdk.chat",
    #    "object": "BaseChat",
    #    "method": "append",
    #    "wrapper_method": task_wrapper,
    #    "span_handler": "xai_handler",
    #    "output_processor": INFERENCE
    #},
    # Client methods that create chat instances
    #{
    #    "package": "xai_sdk.sync.client",
    #    "object": "Client",
    #    "method": "chat",
    ##    "wrapper_method": task_wrapper,
    #    "span_handler": "xai_handler",
    #    "output_processor": INFERENCE
    #},
    #{
    #    "package": "xai_sdk.aio.client",
    #    "object": "Client",
    #    "method": "chat",
    #    "wrapper_method": atask_wrapper,
    #   "span_handler": "xai_handler",
     #   "output_processor": INFERENCE
    #}
]