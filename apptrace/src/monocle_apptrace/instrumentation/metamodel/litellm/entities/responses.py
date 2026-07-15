from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.litellm import (
    _helper,
)
from monocle_apptrace.instrumentation.metamodel.openai.openai_stream_processor import (
    OpenAIStreamProcessor,
)
from monocle_apptrace.instrumentation.common.utils import (
    get_error_message,
    resolve_from_alias,
)


def _is_streaming(kwargs):
    return bool((kwargs.get("response_api_optional_request_params") or {}).get("stream"))


def process_stream(to_wrap, response, span_processor):
    # Responses API streams are SSE response.* events — the OpenAI-format
    # event processor handles them (LiteLLMStreamProcessor is chunk-shaped).
    processor = OpenAIStreamProcessor()
    return processor.process_stream(to_wrap, response, span_processor)


RESPONSES = {
    "type": SPAN_TYPES.INFERENCE,
    "subtype": lambda arguments: _helper.responses_inference_type(arguments),
    "is_auto_close": lambda kwargs: not _is_streaming(kwargs),
    "response_processor": process_stream,
    "attributes": [
        [
            {
                "_comment": "provider type, name, inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: "inference."
                + (arguments["kwargs"].get("custom_llm_provider") or "generic"),
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: _helper.extract_provider_name(
                    (arguments["kwargs"].get("litellm_params") or {}).get("api_base")
                ),
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: (
                    arguments["kwargs"].get("litellm_params") or {}
                ).get("api_base"),
            },
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["kwargs"], ["model"]
                ),
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: "model.llm."
                + resolve_from_alias(arguments["kwargs"], ["model"]),
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
                    "accessor": lambda arguments: _helper.extract_responses_input(
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
                    "accessor": lambda arguments: get_error_message(arguments)
                },
                {
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_responses_output(arguments),
                }
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
                    "_comment": "finish reason from the Responses API result",
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_responses_finish_reason(arguments)
                },
                {
                    "_comment": "finish type mapped from finish reason",
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(
                        _helper.extract_responses_finish_reason(arguments))
                }
            ],
        },
    ],
}
