from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.mistral.entities.inference import MISTRAL_INFERENCE

MISTRAL_METHODS = [
    {
        "package": "mistralai.chat",          # where Chat is defined
        "object": "Chat",                     # class name
        "method": "complete",                 # the sync method
        "span_handler": "non_framework_handler",
        "wrapper_method": task_wrapper,
        "output_processor": MISTRAL_INFERENCE
    }
]

 


