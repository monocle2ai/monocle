from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.botocore import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import (get_error_message, get_llm_type, get_status,)


def process_stream(to_wrap, response, span_processor):
    """Wrap Bedrock streaming responses and compute a synthetic final result for span hydration."""
    if not isinstance(response, dict) or "stream" not in response:
        span_processor(response)
        return response

    original_stream = response.get("stream")
    if original_stream is None:
        span_processor(response)
        return response

    def _wrapped_stream():
        has_tool_use = False
        tool_name = None
        text_parts = []
        finish_reason = None
        usage = None

        try:
            for chunk in original_stream:
                try:
                    if "contentBlockStart" in chunk:
                        start = chunk["contentBlockStart"].get("start", {})
                        tool_use = start.get("toolUse")
                        if tool_use:
                            has_tool_use = True
                            tool_name = tool_use.get("name")

                    if "contentBlockDelta" in chunk:
                        delta = chunk["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            text_parts.append(delta["text"])

                    if "messageStop" in chunk:
                        finish_reason = chunk["messageStop"].get("stopReason", finish_reason)

                    if "metadata" in chunk:
                        metadata = chunk.get("metadata", {})
                        usage = metadata.get("usage", usage)
                except Exception:
                    # Best-effort parsing; never interfere with stream consumption.
                    pass

                yield chunk
        finally:
            if has_tool_use and finish_reason in (None, "end_turn"):
                finish_reason_for_span = "tool_use"
            else:
                finish_reason_for_span = finish_reason or "end_turn"

            content = []
            if text_parts:
                content.append({"text": "".join(text_parts)})
            if has_tool_use and tool_name:
                content.append({"toolUse": {"name": tool_name, "input": {}}})

            synthetic_result = {
                "stopReason": finish_reason_for_span,
                "output": {"message": {"content": content}},
            }
            if usage:
                synthetic_result["usage"] = usage

            span_processor(synthetic_result)

    response["stream"] = _wrapped_stream()
    return response


INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "subtype": lambda arguments: _helper.agent_inference_type(arguments),
    "attributes": [
        [
            {
                "_comment": "provider type  , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: 'inference.'+(get_llm_type(arguments['instance']) or 'generic')
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: arguments['instance'].meta.endpoint_url
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: _helper.extract_provider_name(arguments['instance'])
            }
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: _helper.resolve_from_alias(arguments['kwargs'],
                                                                         ['EndpointName', 'modelId']) or _helper.get_model(arguments['kwargs'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.llm.' + (_helper.resolve_from_alias(arguments['kwargs'],
                                                                                        ['EndpointName', 'modelId']) or _helper.get_model(arguments['kwargs']))
            }
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
        ]
    ],
    "events": [
        {"name": "data.input",
         "attributes": [
             {
                 "_comment": "this is instruction and user query to LLM",
                 "attribute": "input",
                 "accessor": lambda arguments: _helper.extract_messages(arguments['kwargs'])
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
                    "_comment": "this is response from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(arguments)
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(arguments['result'],
                                                                                        arguments['instance'])
                },
                {
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(arguments)
                },
                {
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(
                        _helper.extract_finish_reason(arguments)
                    )
                }
            ]
        }
    ]
}


INFERENCE_STREAM = {
    **INFERENCE,
    "is_auto_close": lambda kwargs: False,
    "response_processor": process_stream,
}
