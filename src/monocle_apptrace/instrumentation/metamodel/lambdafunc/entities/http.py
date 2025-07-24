from monocle_apptrace.instrumentation.metamodel.lambdafunc import _helper
LAMBDA_HTTP_PROCESSOR = {
    "type": "http.process",
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
            {
                "_comment": "request method, request URI",
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
