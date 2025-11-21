from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.mistral import _helper
from monocle_apptrace.instrumentation.common.utils import get_error_message, resolve_from_alias

MISTRAL_INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "subtype": lambda arguments: _helper.agent_inference_type(arguments),
    "attributes": [
        [
            {
                "_comment": "provider type ,name , deployment , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: 'inference.mistral'
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: "mistral"
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: "https://api.mistral.ai"
            }
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.llm.' + resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            }
        ],
        [
            {
                "_comment": "Tool name when finish_type is tool_call",
                "attribute": "name",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.extract_tool_name(arguments),
            },
            {
                "_comment": "Tool type when finish_type is tool_call", 
                "attribute": "type",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.extract_tool_type(arguments),
            },
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is instruction and user query to LLM",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_messages(arguments['kwargs'])
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments)
                },
                {
                    "_comment": "this is result from LLM, works for streaming and non-streaming",
                    "attribute": "response",
                    "accessor": lambda arguments: (
                        # Handle streaming: combine chunks if result is iterable and doesn't have 'choices'
                        _helper.extract_assistant_message(
                            {"result": list(arguments["result"])}
                            if hasattr(arguments.get("result"), "__iter__") and not hasattr(arguments.get("result"), "choices")
                            else arguments
                        )
                    )
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM, includes token counts",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(
                        arguments.get("result"),
                        include_token_counts=True  # new flag for streaming handling
                    )
                },
                {
                    "_comment": "finish reason from Anthropic response",
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(arguments)
                },
                {
                    "_comment": "finish type mapped from finish reason",
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(_helper.extract_finish_reason(arguments))
                }
            ]
        }
    ]
}
