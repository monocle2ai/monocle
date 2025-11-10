import logging
import random
import time
from types import SimpleNamespace
from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.openai import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import (
    get_error_message,
    patch_instance_method,
    resolve_from_alias,
)
from monocle_apptrace.instrumentation.common.constants import PROVIDER_BASE_URLS

logger = logging.getLogger(__name__)


def _process_stream_item(item, state):
    """Process a single stream item and update state."""
    try:
        if (
            hasattr(item, "type")
            and isinstance(item.type, str)
            and item.type.startswith("response.")
        ):
            if state["waiting_for_first_token"]:
                state["waiting_for_first_token"] = False
                state["first_token_time"] = time.time_ns()
            if item.type == "response.output_text.delta":
                state["accumulated_response"] += item.delta
            if item.type == "response.completed":
                state["stream_closed_time"] = time.time_ns()
                if hasattr(item, "response") and hasattr(item.response, "usage"):
                    state["token_usage"] = item.response.usage
        elif (
            hasattr(item, "choices")
            and item.choices
            and item.choices[0].delta
            and item.choices[0].delta.content
        ):
            if hasattr(item.choices[0].delta, "role") and item.choices[0].delta.role:
                state["role"] = item.choices[0].delta.role
            if state["waiting_for_first_token"]:
                state["waiting_for_first_token"] = False
                state["first_token_time"] = time.time_ns()

            state["accumulated_response"] += item.choices[0].delta.content
        elif (
            hasattr(item, "object")
            and item.object == "chat.completion.chunk"
            and item.usage
        ):
            # Handle the case where the response is a chunk
            state["token_usage"] = item.usage
            state["stream_closed_time"] = time.time_ns()
            # Capture finish_reason from the chunk
            if (
                hasattr(item, "choices")
                and item.choices
                and len(item.choices) > 0
                and hasattr(item.choices[0], "finish_reason")
                and item.choices[0].finish_reason
            ):
                finish_reason = item.choices[0].finish_reason
                state["finish_reason"] = finish_reason

    except Exception as e:
        logger.warning(
            "Warning: Error occurred while processing stream item: %s",
            str(e),
        )
    finally:
        state["accumulated_temp_list"].append(item)


def _create_span_result(state, stream_start_time):
    # extract tool calls from the accumulated_temp_list
    # this can only be done when all the streaming is complete.
    for item in state["accumulated_temp_list"]:
        try:
            if (
                item.choices
                and isinstance(item.choices, list)
                and hasattr(item.choices[0], "delta")
                and hasattr(item.choices[0].delta, "tool_calls")
                and item.choices[0].delta.tool_calls
                and item.choices[0].delta.tool_calls[0].id
                and item.choices[0].delta.tool_calls[0].function
            ):
                state["tools"] = state.get("tools", [])
                state["tools"].append(
                    {
                        "id": item.choices[0].delta.tool_calls[0].id,
                        "name": item.choices[0].delta.tool_calls[0].function.name,
                        "arguments": item.choices[0]
                        .delta.tool_calls[0]
                        .function.arguments,
                    }
                )
            if (item.choices and item.choices[0].finish_reason):
                state["finish_reason"] = item.choices[0].finish_reason
        except Exception as e:
            logger.warning(
                "Warning: Error occurred while processing tool calls: %s",
                str(e),
            )

    """Create the span result object."""
    return SimpleNamespace(
        type="stream",
        timestamps={
            "role": state["role"],
            "data.input": int(stream_start_time),
            "data.output": int(state["first_token_time"]),
            "metadata": int(state["stream_closed_time"] or time.time_ns()),
        },
        output_text=state["accumulated_response"],
        tools=state["tools"] if "tools" in state else None,
        usage=state["token_usage"],
        finish_reason=state["finish_reason"],
    )

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
    stream_start_time = time.time_ns()

    # Shared state for both sync and async processing
    state = {
        "waiting_for_first_token": True,
        "first_token_time": stream_start_time,
        "stream_closed_time": None,
        "accumulated_response": "",
        "token_usage": None,
        "accumulated_temp_list": [],
        "finish_reason": None,
        "role": "assistant",
    }

    if to_wrap and hasattr(response, "__iter__"):
        original_iter = response.__iter__

        def new_iter(self):
            for item in original_iter():
                _process_stream_item(item, state)
                yield item

            if span_processor:
                ret_val = _create_span_result(state, stream_start_time)
                span_processor(ret_val)

        patch_instance_method(response, "__iter__", new_iter)

    if to_wrap and hasattr(response, "__aiter__"):
        original_iter = response.__aiter__

        async def new_aiter(self):
            async for item in original_iter():
                _process_stream_item(item, state)
                yield item

            if span_processor:
                ret_val = _create_span_result(state, stream_start_time)
                span_processor(ret_val)

        patch_instance_method(response, "__aiter__", new_aiter)


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
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["kwargs"],
                    ["model", "model_name", "endpoint_name", "deployment_name"],
                ),
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: "model.llm."
                + resolve_from_alias(
                    arguments["kwargs"],
                    ["model", "model_name", "endpoint_name", "deployment_name"],
                ),
            },
        ],
        [
            {
                "_comment": "Tool name when finish_type is tool_call",
                "attribute": "name",
                "accessor": lambda arguments: _helper.extract_tool_name(arguments),
            },
            {
                "_comment": "Tool type when finish_type is tool_call", 
                "attribute": "type",
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
                    "accessor": lambda arguments: _helper.extract_messages(
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
