from monocle_apptrace.instrumentation.common.wrapper import task_wrapper

FLASK_METHODS = [
    {
        "package": "flask.app",
        "object": "Flask",
        "method": "wsgi_app",
        "span_name": "Flask.wsgi_app",
        "wrapper_method": task_wrapper,
        "span_handler": "flask_handler",
        "skip_span": True
    }
]