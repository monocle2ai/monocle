from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.xai.entities.inference import (
    INFERENCE,
)

XAI_METHODS = [
    # Sync Chat methods
    {
        "package": "xai_sdk.sync.chat",
        "object": "Chat",
        "method": "sample",
        "wrapper_method": task_wrapper,
        "span_handler": "xai_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "xai_sdk.sync.chat",
        "object": "Chat", 
        "method": "sample_batch",
        "wrapper_method": task_wrapper,
        "span_handler": "xai_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "xai_sdk.sync.chat",
        "object": "Chat",
        "method": "stream",
        "wrapper_method": task_wrapper,
        "span_handler": "xai_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "xai_sdk.sync.chat",
        "object": "Chat",
        "method": "stream_batch",
        "wrapper_method": task_wrapper,
        "span_handler": "xai_handler",
        "output_processor": INFERENCE
    },
    
    # Async Chat methods
    {
        "package": "xai_sdk.aio.chat",
        "object": "Chat",
        "method": "sample",
        "wrapper_method": atask_wrapper,
        "span_handler": "xai_handler", 
        "output_processor": INFERENCE
    },
    {
        "package": "xai_sdk.aio.chat",
        "object": "Chat",
        "method": "sample_batch",
        "wrapper_method": atask_wrapper,
        "span_handler": "xai_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "xai_sdk.aio.chat",
        "object": "Chat",
        "method": "stream",
        "wrapper_method": atask_wrapper,
        "span_handler": "xai_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "xai_sdk.aio.chat",
        "object": "Chat",
        "method": "stream_batch",
        "wrapper_method": atask_wrapper,
        "span_handler": "xai_handler",
        "output_processor": INFERENCE
    },
]
