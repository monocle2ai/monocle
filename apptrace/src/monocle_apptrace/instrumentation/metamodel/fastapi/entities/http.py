from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.fastapi import _helper

FASTAPI_HTTP_PROCESSOR = {
    "type": SPAN_TYPES.HTTP_PROCESS,
    "attributes": [
        [
            {
                "attribute": "method",
                "accessor": lambda arguments: _helper.get_method(arguments['args'][0])
            },
            {
                "attribute": "route",
                "accessor": lambda arguments: _helper.get_route(arguments['args'][0])
            },
        {
                "attribute": "url",
                "accessor": lambda arguments: _helper.get_url(arguments['args'][0])
            },
        ]
    ]
}

FASTAPI_RESPONSE_PROCESSOR = {
    "type": SPAN_TYPES.GENERIC,
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "params",
                    "accessor": lambda arguments: _helper.get_params(arguments['args'][0])
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: _helper.extract_status(arguments)
                },
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_response(arguments['instance'])
                }
            ]
        }
    ]
}