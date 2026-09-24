from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.httpx.entities.http import HTTPX_HTTP_PROCESSOR

# Every httpx call funnels through `send` -- get, post, request, and the SDKs
# built on them -- so wrapping it once covers them all.
HTTPX_METHODS = [
    {
        "package": "httpx",
        "object": "Client",
        "method": "send",
        "wrapper_method": task_wrapper,
        "span_handler": "httpx_handler",
        "output_processor": HTTPX_HTTP_PROCESSOR
    },
    {
        "package": "httpx",
        "object": "AsyncClient",
        "method": "send",
        "wrapper_method": atask_wrapper,
        "span_handler": "httpx_handler",
        "output_processor": HTTPX_HTTP_PROCESSOR
    }
]
