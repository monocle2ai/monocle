from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from monocle_apptrace.instrumentation.metamodel.aiohttp.entities.http import AIO_HTTP_PROCESSOR

AIOHTTP_METHODS = [
    {
        "package": "aiohttp.web_app",
        "object": "Application",
        "method": "_handle",
        "wrapper_method": atask_wrapper,
        "span_handler": "aiohttp_handler",
        "output_processor": AIO_HTTP_PROCESSOR
    }
]