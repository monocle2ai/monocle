import logging
from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.xai import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import (
    get_error_message,
    resolve_from_alias,
)

logger = logging.getLogger(__name__)


def extract_vector_input(vector_input: dict):
    """Extract input from xAI embedding request"""
    if 'input' in vector_input:
        return vector_input['input']
    return ""


def extract_vector_output(vector_output):
    """Extract embedding output from xAI embedding response"""
    try:
        if hasattr(vector_output, 'data') and len(vector_output.data) > 0:
            return vector_output.data[0].embedding
    except Exception:
        pass
    return ""


RETRIEVAL = {
    "span_name": "xai.embedding",
    "span_type": SPAN_TYPES.RETRIEVAL,
    "attributes": [
        [
            {
                "_comment": "Provider Name",
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
        ],
        [
            {
                "_comment": "Embedding Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["kwargs"],
                    ["model", "model_name"],
                ),
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: "model.embedding."
                + resolve_from_alias(
                    arguments["kwargs"],
                    ["model", "model_name"],
                ),
            },
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is input to embedding model",
                    "attribute": "input",
                    "accessor": lambda arguments: extract_vector_input(
                        arguments["kwargs"]
                    ),
                }
            ],
        },
        {
            "name": "data.output", 
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
                {
                    "_comment": "this is embedding vector output",
                    "attribute": "response",
                    "accessor": lambda arguments: extract_vector_output(
                        arguments["result"]
                    ),
                },
            ],
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from embedding model",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(
                        arguments["result"]
                    ),
                },
            ],
        },
    ],
}