from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.flask import _helper
FLASK_HTTP_PROCESSOR = {
    "type": SPAN_TYPES.HTTP_PROCESS,
    "attributes": [
        [
            {
                "_comment": "request method, request URI",
                "attribute": "method",
                "accessor": lambda arguments: _helper.get_method(arguments['args'])
            },
            {
                "_comment": "request method, request URI",
                "attribute": "route",
                "accessor": lambda arguments: _helper.get_route(arguments['args'])
            },
        ]
    ]
}

FLASK_RESPONSE_PROCESSOR = {
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "route params",
                    "attribute": "params",
                    "accessor": lambda arguments: _helper.get_params(arguments['args'])
                }
         ]
         },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "status from HTTP response",
                    "attribute": "error_code",
                    "accessor": lambda arguments: _helper.extract_status(arguments)
                },
                {
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_response(arguments['instance'])
                }
            ]
        }
    ]
}