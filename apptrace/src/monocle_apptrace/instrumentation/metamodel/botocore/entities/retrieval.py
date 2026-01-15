from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.botocore import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import (get_error_message, get_llm_type)

RETRIEVAL = {
    "type": SPAN_TYPES.RETRIEVAL,
    "attributes": [
        [
            {
                "_comment": "provider type, inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: 'retrieval.aws_bedrock'
            },
            {
                "attribute": "endpoint",
                "accessor": lambda arguments: arguments['instance'].meta.endpoint_url
            }
        ],
        [
            {
                "_comment": "Knowledge base or retrieval source",
                "attribute": "name",
                "accessor": lambda arguments: _helper.extract_retrieval_source(arguments['kwargs'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'vectorstore.' + _helper.extract_retrieval_source(arguments['kwargs'])
            }
        ]
    ],
    "events": [
        {"name": "data.input",
         "attributes": [
             {
                 "_comment": "retrieval query input",
                 "attribute": "input",
                 "accessor": lambda arguments: _helper.extract_retrieval_query(arguments['kwargs'])
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
                    "_comment": "retrieval response with citations",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_retrieval_response(arguments)
                }
            ]
        }
    ]
}
