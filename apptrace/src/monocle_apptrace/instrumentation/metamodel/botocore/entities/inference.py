from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.botocore import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import (get_error_message, get_llm_type, get_status,)
INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "subtype": lambda arguments: _helper.agent_inference_type(arguments),
    "attributes": [
        [
            {
                "_comment": "provider type  , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: 'inference.'+(get_llm_type(arguments['instance']) or 'generic')
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: arguments['instance'].meta.endpoint_url
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: _helper.extract_provider_name(arguments['instance'])
            }
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: _helper.resolve_from_alias(arguments['kwargs'],
                                                                         ['EndpointName', 'modelId'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.llm.' + _helper.resolve_from_alias(arguments['kwargs'],
                                                                                        ['EndpointName', 'modelId'])
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
        {"name": "data.input",
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
                    "_comment": "this is response from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(arguments)
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(arguments['result'],
                                                                                        arguments['instance'])
                },
                {
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(arguments)
                },
                {
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(
                        _helper.extract_finish_reason(arguments)
                    )
                }
            ]
        }
    ]
}
