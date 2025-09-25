import logging
import time
from types import SimpleNamespace
from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.xai import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import (
    get_error_message,
    resolve_from_alias,
)

logger = logging.getLogger(__name__)


def _process_xai_response(response, start_time):
    """Process xAI response and create span result."""
    try:
        # Handle xAI response object
        result_response = SimpleNamespace()
        result_response.role = "assistant"
        
        # Extract content from different response formats
        if hasattr(response, "content"):
            result_response.output_text = response.content
        elif hasattr(response, "text"):
            result_response.output_text = response.text
        elif isinstance(response, str):
            result_response.output_text = response
        else:
            result_response.output_text = str(response)
        
        # Add timing information if available
        result_response.stream_start_time = start_time
        result_response.stream_closed_time = time.time_ns()
        
        # Handle token usage if available
        if hasattr(response, "usage"):
            result_response.token_usage = response.usage
        
        return result_response
    except Exception as e:
        logger.warning(f"Error processing xAI response: {str(e)}")
        return response


INFERENCE = {
    "span_name": "xai.sample",
    "type": SPAN_TYPES.INFERENCE,
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
            {
                "_comment": "Inference Type",
                "attribute": "inference_type",
                "accessor": lambda arguments: _helper.get_inference_type(
                    arguments["instance"]
                ),
            },
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["kwargs"],
                    ["model", "model_name"],
                ),
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: "model.llm."
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
                    "_comment": "this is instruction and user query to LLM",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_messages(arguments),
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
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(arguments),
                },
            ],
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(
                        arguments["result"]
                    ),
                },
                {
                    "_comment": "finish reason from xAI response",
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(
                        arguments
                    ),
                },
                {
                    "_comment": "finish type mapped from finish reason",
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(
                        _helper.extract_finish_reason(arguments)
                    ),
                },
                {
                    "attribute": "inference_sub_type",
                    "accessor": lambda arguments: _helper.agent_inference_type(arguments)
                }
            ],
        },
    ],
}
