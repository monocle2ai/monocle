import logging

from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.openai.openai_stream_processor import (
    OpenAIStreamProcessor,
)
from monocle_apptrace.instrumentation.metamodel.openai import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import (
    get_error_message,
    resolve_from_alias,
)

logger = logging.getLogger(__name__)

# Registry mapping client detection functions → entity_type
CLIENT_ENTITY_MAP = {
    "deepseek": "inference.deepseek",
    # add more clients in future
}

def get_entity_type(response, helper=None):
    for client_name, entity in CLIENT_ENTITY_MAP.items():
        check_fn = globals().get(f"is_{client_name}_client")
        if check_fn and check_fn(response):
            return entity

    # fallback to helper if available
    if helper and hasattr(helper, "get_inference_type"):
        return "inference." + helper.get_inference_type(response)

    # default fallback
    return "inference.openai"


def process_stream(to_wrap, response, span_processor):
    processor = OpenAIStreamProcessor()
    processor.process_stream(to_wrap, response, span_processor)
    

INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "subtype": lambda arguments: _helper.agent_inference_type(arguments),
    "is_auto_close": lambda kwargs: kwargs.get("stream", False) is False,
    "response_processor": process_stream,
    "attributes": [
        [
            {
                "_comment": "provider type ,name , deployment , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: "inference."
                + (_helper.get_inference_type(arguments["instance"]))
                or "openai",
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: _helper.extract_provider_name(
                    arguments["instance"]
                ),
            },
            {
                "attribute": "deployment",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["instance"].__dict__,
                    [
                        "engine",
                        "azure_deployment",
                        "deployment_name",
                        "deployment_id",
                        "deployment",
                    ],
                ),
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["instance"].__dict__,
                    ["azure_endpoint", "api_base", "endpoint"],
                )
                or _helper.extract_inference_endpoint(arguments["instance"]),
            },
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: _helper.extract_model_name(arguments),
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: _helper.extract_model_type(arguments),
            },
        ],
        [
            {
                "_comment": "Tool name when finish_type is tool_call",
                "attribute": "name",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.extract_tool_name(arguments),
            },
            {
                "_comment": "Tool type when finish_type is tool_call", 
                "attribute": "type",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.extract_tool_type(arguments),
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
                    "accessor": lambda arguments: _helper.extract_messages_from_arguments(
                        arguments
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
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(
                        arguments,
                    ),
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
                    "_comment": "finish reason from OpenAI response",
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
                }
            ],
        },
    ],
}
