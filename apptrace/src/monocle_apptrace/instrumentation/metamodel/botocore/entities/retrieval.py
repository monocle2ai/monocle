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
                "_comment": "vector store name and type",
                "attribute": "name",
                "accessor": lambda arguments: _helper.extract_vector_name(arguments)
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'vectorstore.' + _helper.extract_vector_name(arguments)
            },
        ],
        [
            {
                "_comment": "embedding model name and type",
                "attribute": "name",
                "accessor": lambda arguments: _helper.extract_embedding_model(arguments)
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.embedding.' + _helper.extract_embedding_model(arguments)
            }
        ],
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
