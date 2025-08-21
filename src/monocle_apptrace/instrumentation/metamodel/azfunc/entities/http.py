from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.azfunc import _helper
AZFUNC_HTTP_PROCESSOR = {
    "type": SPAN_TYPES.HTTP_PROCESS,
    "attributes": [
        [
            {
                "_comment": "request method, request URI",
                "attribute": "method",
                "accessor": lambda arguments: _helper.get_method(arguments['kwargs'])
            },
            {
                "_comment": "request method, request URI",
                "attribute": "route",
                "accessor": lambda arguments: _helper.get_route(arguments['kwargs'])
            },
            {
                "_comment": "request method, request URI",
                "attribute": "body",
                "accessor": lambda arguments: _helper.get_body(arguments['kwargs'])
            },
            {
                "_comment": "request function name",
                "attribute": "function_name",
                "accessor": lambda arguments: _helper.get_function_name(arguments['kwargs'])
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "route params",
                    "attribute": "params",
                    "accessor": lambda arguments: _helper.get_params(arguments['kwargs'])
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
