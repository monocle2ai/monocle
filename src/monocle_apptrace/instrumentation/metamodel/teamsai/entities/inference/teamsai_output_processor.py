from monocle_apptrace.instrumentation.metamodel.teamsai import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import get_llm_type
TEAMAI_OUTPUT_PROCESSOR = {
    "type": "inference.framework",
    "attributes": [
        [
            {
                "_comment": "provider type, name, deployment",
                "attribute": "type",
                "accessor": lambda arguments: 'inference.' + (get_llm_type(arguments['instance']._client) or 'generic')
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: _helper.extract_provider_name(arguments['instance'])
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: _helper.extract_inference_endpoint(arguments['instance'])
            }
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: arguments["instance"]._options.default_model if hasattr(arguments["instance"], "_options") else "unknown"
            },
            {
                "_comment": "LLM Model",
                "attribute": "type",
                "accessor": lambda arguments: 'model.llm.'+ arguments["instance"]._options.default_model if hasattr(arguments["instance"], "_options") else "unknown"
            },
            {
                "attribute": "is_streaming",
                "accessor": lambda arguments: arguments["instance"]._options.stream if hasattr(arguments["instance"], "_options") else False
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "_comment": "input to Teams AI",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": _helper.capture_input
                }
            ]
        },
        {
            "name": "data.output",
            "_comment": "output from Teams AI",
            "attributes": [
                {
                    "attribute": "status",
                    "accessor": lambda arguments: _helper.get_status(arguments)
                },
                {
                    "attribute": "status_code",
                    "accessor": lambda arguments: _helper.get_status_code(arguments)
                },
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.get_response(arguments)
                },
                {
                    "attribute": "check_status",
                    "accessor": lambda arguments: _helper.check_status(arguments)
                }
            ]
        },
    ]
}