from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.httpx import _helper

HTTPX_HTTP_PROCESSOR = {
    "type": SPAN_TYPES.HTTP_SEND,
    "attributes": [
        [
            {
                "_comment": "request method",
                "attribute": "method",
                "accessor": lambda arguments: _helper.get_method(arguments)
            },
            {
                "_comment": "request URI",
                "attribute": "URL",
                "accessor": lambda arguments: _helper.get_route(arguments)
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "route params",
                    "attribute": "http.params",
                    "accessor": lambda arguments: _helper.get_params(arguments)
                },
                {
                    "_comment": "route body",
                    "attribute": "body",
                    "accessor": lambda arguments: _helper.get_body(arguments)
                },
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "status from HTTP response",
                    "attribute": "status",
                    "accessor": lambda arguments: _helper.extract_status(arguments['result'])
                },
                {
                    "_comment": "body of the HTTP response",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_response(arguments['result'])
                }
            ]
        }
    ]
}
