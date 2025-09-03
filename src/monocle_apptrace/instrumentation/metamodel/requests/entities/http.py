from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.requests import _helper
REQUEST_HTTP_PROCESSOR = {
    "type": SPAN_TYPES.HTTP_SEND,
    "attributes": [
        [
            {
                "_comment": "request method, request URI",
                "attribute": "method",
                "accessor": lambda arguments: _helper.get_method(arguments['kwargs'])
            },
            {
                "_comment": "request method, request URI",
                "attribute": "URL",
                "accessor": lambda arguments: _helper.get_route(arguments['kwargs'])
            }

        ]
    ],
    "events": [
        {"name": "data.input",
         "attributes": [
             {
                 "_comment": "route params",
                 "attribute": "http.params",
                 "accessor": lambda arguments: _helper.get_params(arguments['kwargs'])
             },
             {
                 "_comment": "route body",
                 "attribute": "body",
                 "accessor": lambda arguments: _helper.get_body(arguments['kwargs'])
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
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_response(arguments['result'])
                }
            ]
        }
    ]
}