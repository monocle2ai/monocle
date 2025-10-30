from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper,task_wrapper
from monocle_apptrace.instrumentation.metamodel.fastapi.entities.http import FASTAPI_HTTP_PROCESSOR, FASTAPI_RESPONSE_PROCESSOR

FASTAPI_METHODS = [
    {
        "package": "fastapi",
        "object": "FastAPI",
        "method": "__call__",
        "wrapper_method": atask_wrapper,
        "span_name": "fastapi.request",
        "span_handler": "fastapi_handler",
        "output_processor": FASTAPI_HTTP_PROCESSOR,
    },
    {
        "package": "starlette.responses",
        "object": "Response",
        "method": "__call__",
        "span_name": "fastapi.response",
        "wrapper_method": task_wrapper,
        "span_handler": "fastapi_response_handler",
        "output_processor": FASTAPI_RESPONSE_PROCESSOR
    }
]