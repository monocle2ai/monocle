from monocle_apptrace.instrumentation.metamodel.fastapi import _helper

FASTAPI_HTTP_PROCESSOR = {
    "type": "http.process",
    "attributes": [
        [
            {
                "_comment": "request method, request URI",
                "attribute": "method",
                "accessor": lambda arguments: _helper.get_method(arguments['args'])
            },
            {
                "_comment": "request route",
                "attribute": "route",
                "accessor": lambda arguments: _helper.get_route(arguments['args'])
            },
            {
                "_comment": "request body",
                "attribute": "body",
                "accessor": lambda arguments: _helper.get_body(arguments['args'])
            },
        ]
    ],
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
                    "attribute": "status",
                    "accessor": lambda arguments: _helper.extract_status(arguments['instance'])
                },
                {
                    "_comment": "response content",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_response(arguments['instance'])
                }
            ]
        }
    ]
}
