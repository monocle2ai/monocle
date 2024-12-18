from monocle_apptrace.instrumentation.metamodel.botocore import (
    _helper,
)

INFERENCE = {
    "type": "inference",
    "attributes": [
        [
            {
                "_comment": "provider type  , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: 'inference.aws_sagemaker'
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: arguments['instance'].meta.endpoint_url
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
                    "_comment": "this is response from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(arguments['result'])
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
