from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.gemini import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import get_error_message

INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "attributes": [
        [
            {
                "_comment": "provider type  , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: 'inference.vertexai' if hasattr(arguments['instance'],"vertexai") and arguments['instance'].vertexai else 'inference.gemini'
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: _helper.extract_inference_endpoint(arguments['instance'])
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: 'gcp'
            }
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: _helper.resolve_from_alias(arguments['kwargs'],
                                                                         ['model'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.llm.' + _helper.resolve_from_alias(arguments['kwargs'],
                                                                                        ['model'])
            }
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
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(arguments['result'], arguments['instance'])
                },
                {
                    "_comment": "finish reason from Gemini response",
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(arguments)
                },
                {
                    "_comment": "finish type mapped from finish reason",
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(
                        _helper.extract_finish_reason(arguments)
                    )
                }
            ]
        }


    ]
}
