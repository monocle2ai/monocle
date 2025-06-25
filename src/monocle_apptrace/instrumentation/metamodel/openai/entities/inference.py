import logging
import random
import time
from types import SimpleNamespace
from monocle_apptrace.instrumentation.metamodel.openai import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import (
    patch_instance_method,
    resolve_from_alias,
    get_status,
    get_exception_status_code,
    get_status_code,
)

logger = logging.getLogger(__name__)


def _process_stream_item(item, state):
    """Process a single stream item and update state."""
    try:
        if (
            item.choices
            and item.choices[0].delta
            and item.choices[0].delta.content
        ):
            if hasattr(item.choices[0].delta, "role") and item.choices[0].delta.role:
                state["role"] = item.choices[0].delta.role
            if state["waiting_for_first_token"]:
                state["waiting_for_first_token"] = False
                state["first_token_time"] = time.time_ns()

            state["accumulated_response"] += item.choices[0].delta.content
        elif item.object == "chat.completion.chunk" and item.usage:
            # Handle the case where the response is a chunk
            state["token_usage"] = item.usage
            state["stream_closed_time"] = time.time_ns()

    except Exception as e:
        logger.warning(
            "Warning: Error occurred while processing stream item: %s",
            str(e),
        )
    finally:
        state["accumulated_temp_list"].append(item)


def _create_span_result(state, stream_start_time):
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
        usage=state["token_usage"],
    )


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
    "type": "inference",
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
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(
                        arguments,
                    ),
                },
                {
                    "attribute": "status",
                    "accessor": lambda arguments: get_status(arguments)
                },
                {
                    "attribute": "status_code",
                    "accessor": lambda arguments: get_status_code(arguments)
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
                }
            ],
        },
    ],
}
