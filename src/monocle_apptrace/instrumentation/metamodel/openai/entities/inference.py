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
    get_exception_status_code
)

logger = logging.getLogger(__name__)


def process_stream(to_wrap, response, span_processor):
    waiting_for_first_token = True
    stream_start_time = time.time_ns()
    first_token_time = stream_start_time
    stream_closed_time = None
    accumulated_response = ""
    token_usage = None
    accumulated_temp_list = []

    if to_wrap and hasattr(response, "__iter__"):
        original_iter = response.__iter__

        def new_iter(self):
            nonlocal waiting_for_first_token, first_token_time, stream_closed_time, accumulated_response, token_usage

            for item in original_iter():
                try:
                    if (
                        item.choices
                        and item.choices[0].delta
                        and item.choices[0].delta.content
                    ):
                        if waiting_for_first_token:
                            waiting_for_first_token = False
                            first_token_time = time.time_ns()

                        accumulated_response += item.choices[0].delta.content
                        # token_usage = item.usage
                    elif item.object == "chat.completion.chunk" and item.usage:
                        # Handle the case where the response is a chunk
                        token_usage = item.usage
                        stream_closed_time = time.time_ns()

                except Exception as e:
                    logger.warning(
                        "Warning: Error occurred while processing item in new_iter: %s",
                        str(e),
                    )
                finally:
                    accumulated_temp_list.append(item)
                    yield item

            if span_processor:
                ret_val = SimpleNamespace(
                    type="stream",
                    timestamps={
                        "data.input": int(stream_start_time),
                        "data.output": int(first_token_time),
                        "metadata": int(stream_closed_time or time.time_ns()),
                    },
                    output_text=accumulated_response,
                    usage=token_usage,
                )
                span_processor(ret_val)

        patch_instance_method(response, "__iter__", new_iter)
        
    if to_wrap and hasattr(response, "__aiter__"):
        original_iter = response.__aiter__

        async def new_aiter(self):
            nonlocal waiting_for_first_token, first_token_time, stream_closed_time, accumulated_response, token_usage

            async for item in original_iter():
                try:
                    if (
                        item.choices
                        and item.choices[0].delta
                        and item.choices[0].delta.content
                    ):
                        if waiting_for_first_token:
                            waiting_for_first_token = False
                            first_token_time = time.time_ns()

                        accumulated_response += item.choices[0].delta.content
                        # token_usage = item.usage
                    elif item.object == "chat.completion.chunk" and item.usage:
                        # Handle the case where the response is a chunk
                        token_usage = item.usage
                        stream_closed_time = time.time_ns()

                except Exception as e:
                    logger.warning(
                        "Warning: Error occurred while processing item in new_aiter: %s",
                        str(e),
                    )
                finally:
                    accumulated_temp_list.append(item)
                    yield item

            if span_processor:
                ret_val = SimpleNamespace(
                    type="stream",
                    timestamps={
                        "data.input": int(stream_start_time),
                        "data.output": int(first_token_time),
                        "metadata": int(stream_closed_time or time.time_ns()),
                    },
                    output_text=accumulated_response,
                    usage=token_usage,
                )
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
                    "accessor": lambda arguments: get_exception_status_code(arguments)
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
