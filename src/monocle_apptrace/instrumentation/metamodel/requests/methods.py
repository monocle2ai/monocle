from monocle_apptrace.instrumentation.common.wrapper import task_wrapper

REQUESTS_METHODS = [
    {
        "package": "requests.sessions",
        "object": "Session",
        "method": "request",
        "span_name": "http_requests",
        "wrapper_method": task_wrapper,
        "span_handler":"request_handler",
    }
]