from monocle_apptrace.instrumentation.common.wrapper import task_wrapper

REQUESTS_METHODS = [
    {
        "package": "requests.sessions",
        "object": "Session",
        "method": "request",
        "span_name": "http_requests",
        "wrapper_method": task_wrapper,
        "pre_task_processor": {
            "module": "monocle_apptrace.instrumentation.metamodel.requests._helper",
            "method": "request_pre_processor"
        }
    }
]