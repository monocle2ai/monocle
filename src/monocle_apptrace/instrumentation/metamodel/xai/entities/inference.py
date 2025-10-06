from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.common.utils import get_status_code, resolve_from_alias
from monocle_apptrace.instrumentation.metamodel.xai import _helper

def process_stream(to_wrap, response, span_processor):
    """Process xAI SDK streaming responses"""
    # TODO: Implement streaming support for xAI SDK if needed
    # For now, skip streaming implementation
    pass

INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "is_auto_close": lambda kwargs: True,  # Set to False if streaming supported
    # "response_processor": process_stream,  # Uncomment if streaming supported
    "attributes": [
        [
            # Provider information
            {
                "_comment": "provider type, name, deployment, inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: "inference."
                + (_helper.get_inference_type(arguments["instance"]))
                or "xai",
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
        ],
        [
            # Model information - xAI SDK stores model in the chat instance
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: getattr(arguments["instance"], '_proto', {}).model if hasattr(arguments["instance"], '_proto') else "",
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.llm.' + 'model' if hasattr(arguments["instance"], '_proto') else ""
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
                    "accessor": lambda arguments: _helper.extract_messages_from_instance(
                        arguments["instance"]
                    ),
                },
                {
                    "attribute": "parameters",
                    "accessor": lambda arguments: _helper.get_json_dumps({
                        k: v for k, v in arguments["kwargs"].items() 
                        if k not in ["messages", "prompt"]  # Exclude message content
                    }),
                },
            ],
        },
        {
            "name": "data.output", 
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_status_code(arguments),
                },
                {
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(
                        arguments
                    ),
                },
            ],
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM",
                    "accessor": lambda arguments: _helper.get_json_dumps(_helper.extract_usage(arguments)),
                },
                {
                    "_comment": "finish reason from xAI SDK response",
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
