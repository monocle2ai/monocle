from monocle_apptrace.instrumentation.metamodel.openai import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias

RETRIEVAL = {
    "type": "retrieval",
    "attributes": [
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.embedding.' + resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
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
                     "accessor": lambda arguments: _helper.update_input_span_events(arguments['kwargs'])
                 }
         ]
         },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.update_output_span_events(arguments['result'])
                }
            ]
        }
    ]
}
