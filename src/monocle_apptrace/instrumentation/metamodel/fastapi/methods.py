from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.fastapi.entities.http import FASTAPI_HTTP_PROCESSOR

FASTAPI_METHODS = [
    {
        "package": "starlette.responses",
        "object": "Response",
        "method": "__call__",
        "span_name": "fastapi.request",
        "wrapper_method": atask_wrapper,
        "span_handler": "fastapi_handler",
        "output_processor": FASTAPI_HTTP_PROCESSOR
    }
]
