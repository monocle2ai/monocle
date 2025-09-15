from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.requests.entities.http import REQUEST_HTTP_PROCESSOR

REQUESTS_METHODS = [
    {
        "package": "requests.sessions",
        "object": "Session",
        "method": "request",
        "wrapper_method": task_wrapper,
        "span_handler":"request_handler",
        "output_processor": REQUEST_HTTP_PROCESSOR
    }
]