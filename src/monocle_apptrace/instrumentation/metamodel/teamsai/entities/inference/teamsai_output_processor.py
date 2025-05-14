from monocle_apptrace.instrumentation.metamodel.teamsai import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import get_llm_type
TEAMAI_OUTPUT_PROCESSOR = {
    "type": "inference",
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
                    "attribute": "response",
                    "accessor": lambda arguments: arguments["result"].message.content if hasattr(arguments["result"], "message") else str(arguments["result"])
                }
            ]
        },
        # {
        #     "name": "metadata",
        #     "attributes": [
        #         {
        #             "_comment": "metadata from Teams AI response",
        #             "accessor": lambda arguments: {
        #                 "prompt_tokens": arguments["result"].get("usage", {}).get("prompt_tokens", 0),
        #                 "completion_tokens": arguments["result"].get("usage", {}).get("completion_tokens", 0),
        #                 "total_tokens": arguments["result"].get("usage", {}).get("total_tokens", 0),
        #                 "latency_ms": arguments.get("latency_ms")
        #             }
        #         }
        #     ]
        # }
    ]
}