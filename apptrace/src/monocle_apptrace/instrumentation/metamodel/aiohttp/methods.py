from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from monocle_apptrace.instrumentation.metamodel.aiohttp.entities.http import AIO_HTTP_PROCESSOR
from monocle_apptrace.instrumentation.metamodel.aiohttp._helper import (
    aiohttp_streamresponse_prepare, aiohttp_streamresponse_write_eof,
)

AIOHTTP_METHODS = [
    {
        "package": "aiohttp.web_app",
        "object": "Application",
        "method": "_handle",
        "wrapper_method": atask_wrapper,
        "span_handler": "aiohttp_handler",
        "output_processor": AIO_HTTP_PROCESSOR
    },
    {
        # Streaming trace-return support: sets the x-monocle-traces header before
        # headers are sent. No span/output processor -- pure behavior wrapper,
        # mirrors fastapi's streaming_response_wrapper (calls wrapped(...) directly,
        # never creates a span).
        "package": "aiohttp.web_response",
        "object": "StreamResponse",
        "method": "prepare",
        "wrapper_method": aiohttp_streamresponse_prepare,
    },
    {
        # Streaming trace-return support: writes the trailer just before EOF.
        "package": "aiohttp.web_response",
        "object": "StreamResponse",
        "method": "write_eof",
        "wrapper_method": aiohttp_streamresponse_write_eof,
    },
]