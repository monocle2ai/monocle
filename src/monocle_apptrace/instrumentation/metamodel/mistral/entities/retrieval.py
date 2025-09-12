from monocle_apptrace.instrumentation.metamodel.mistral import _helper
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias

MISTRAL_RETRIEVAL = {
    "type": "embedding",
    "attributes": [
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(arguments['kwargs'], ['model'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.embedding.' + resolve_from_alias(arguments['kwargs'], ['model'])
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "embedding input",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.update_input_span_events(arguments["kwargs"])
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "embedding output summary",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.update_output_span_events(arguments["result"])
                }
            ]
        }
    ]
}
