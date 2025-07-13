from monocle_apptrace.instrumentation.metamodel.haystack import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import get_error_message, get_llm_type

INFERENCE = {
    "type": "inference.framework",
    "attributes": [
        [
            {
                "_comment": "provider type ,name , deployment , inference_endpoint",
                "attribute": "type",
#                "accessor": lambda arguments: 'inference.azure_openai'
                "accessor": lambda arguments: 'inference.' + (get_llm_type(arguments['instance']) or 'generic')

            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: arguments['kwargs']['provider_name']
            },
            {
                "attribute": "deployment",
                "accessor": lambda arguments: _helper.resolve_from_alias(arguments['instance'].__dict__,
                                                                         ['engine', 'azure_deployment',
                                                                          'deployment_name', 'deployment_id',
                                                                          'deployment'])
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: _helper.resolve_from_alias(arguments['instance'].__dict__, ['api_base_url']) or _helper.extract_inference_endpoint(arguments['instance'])
            }
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: _helper.resolve_from_alias(arguments['instance'].__dict__,
                                                                         ['model', 'model_name'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.llm.' + _helper.resolve_from_alias(arguments['instance'].__dict__,
                                                                                        ['model', 'model_name'])
            }
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
                }
            ]
        }
    ]
}
