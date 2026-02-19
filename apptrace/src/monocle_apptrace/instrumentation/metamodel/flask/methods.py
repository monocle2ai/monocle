from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.flask.entities.http import FLASK_HTTP_PROCESSOR, FLASK_RESPONSE_PROCESSOR

FLASK_METHODS = [
    {
        "package": "flask.app",
        "object": "Flask",
        "method": "wsgi_app",
        "wrapper_method": task_wrapper,
        "span_handler": "flask_handler",
        "output_processor": FLASK_HTTP_PROCESSOR,
    },
    {
        "package": "werkzeug.wrappers.response",
        "object": "Response",
        "method": "__call__",
        "wrapper_method": task_wrapper,
        "span_handler": "flask_response_handler",
        "output_processor": FLASK_RESPONSE_PROCESSOR,
    }
]