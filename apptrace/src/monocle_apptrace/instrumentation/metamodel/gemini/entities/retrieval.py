from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.gemini import (
    _helper,
)

RETRIEVAL = {
    "type": SPAN_TYPES.RETRIEVAL,
    "attributes": [
        [
            {
                "_comment": "Embedding Model",
                "attribute": "name",
                "accessor": lambda arguments: _helper.resolve_from_alias(arguments['kwargs'],
                                                                         ['model'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.embedding.' + _helper.resolve_from_alias(arguments['kwargs'],
                                                                                        ['model'])
            }
        ]
    ],
    "events": [
        {
         "name": "data.input",
         "attributes": [
             {
                 "attribute": "input",
                 "accessor": lambda arguments: _helper.update_input_span_events(arguments['kwargs'])
             }
         ]
         },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.update_output_span_events(arguments['result'])
                }
            ]
        }

    ]
}
