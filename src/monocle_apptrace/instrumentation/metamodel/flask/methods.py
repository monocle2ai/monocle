from monocle_apptrace.instrumentation.common.wrapper import task_wrapper

FLASK_METHODS = [
    {
        "package": "flask.app",
        "object": "Flask",
        "method": "wsgi_app",
        "span_name": "Flask.wsgi_app",
        "wrapper_method": task_wrapper,
        "pre_task_action_processor": {
            "module": "monocle_apptrace.instrumentation.metamodel.flask._helper",
            "method": "flask_pre_processor"
        },
        "post_task_action_processor": {
            "module": "monocle_apptrace.instrumentation.metamodel.flask._helper",
            "method": "flask_post_processor"
        },
        "skip_span": True
    }
]