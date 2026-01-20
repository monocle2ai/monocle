from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES, SPAN_SUBTYPES
from monocle_apptrace.instrumentation.metamodel.agentcore import _helper

AGENTCORE_PROCESSOR = {
    "type": SPAN_TYPES.HTTP_PROCESS,
    "attributes": [
        [
            {
                "_comment": "request route",
                "attribute": "route",
                "accessor": lambda arguments: _helper.get_route(arguments['args'])
            },
            {
                "_comment": "request method",
                "attribute": "method",
                "accessor": lambda arguments: "POST"
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "route params",
                    "attribute": "request",
                    "accessor": lambda arguments: _helper.get_inputs(arguments['args'])
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
                    "accessor": lambda arguments: _helper.get_outputs(arguments['result'])
                }
            ]
        }
    ]
}
