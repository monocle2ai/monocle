from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.ollama import _helper
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias

RETRIEVAL = {
    "type": SPAN_TYPES.RETRIEVAL,
    "attributes": [
        [
            {
                "attribute": "type", 
                "accessor": lambda arguments: "retrieval.ollama",
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: _helper.extract_provider_name(
                    arguments["instance"]
                ),
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: _helper.extract_inference_endpoint(
                    arguments["instance"]
                ),
            },
            {
                "attribute": "model_name",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["kwargs"], ["model"]
                ),
            },
        ],
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_vector_input(
                        arguments["kwargs"]
                    ),
                },
                {
                    "attribute": "parameters",
                    "accessor": lambda arguments: resolve_from_alias(
                        arguments["kwargs"],
                        [
                            "truncate",
                            "options", 
                            "keep_alive",
                            "dimensions"
                        ],
                    ),
                },
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_vector_output(
                        arguments["result"]
                    ),
                },
            ],
        },
    ],
}
