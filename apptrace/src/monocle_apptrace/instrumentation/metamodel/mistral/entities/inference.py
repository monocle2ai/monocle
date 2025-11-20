import logging
import time
from types import SimpleNamespace
from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.mistral import _helper
from monocle_apptrace.instrumentation.common.utils import (
    get_error_message, 
    resolve_from_alias,
    patch_instance_method
)

logger = logging.getLogger(__name__)


def _process_stream_item(item, state):
    """Process a single Mistral stream item and update state."""
    try:
        # Mistral streaming uses chunk.data.choices[0].delta.content
        if (
            hasattr(item, "data")
            and hasattr(item.data, "choices")
            and item.data.choices
            and hasattr(item.data.choices[0], "delta")
        ):
            delta = item.data.choices[0].delta
            
            # Handle role assignment
            if hasattr(delta, "role") and delta.role:
                state["role"] = delta.role
            
            # Handle content
            if hasattr(delta, "content") and delta.content is not None:
                if state["waiting_for_first_token"]:
                    state["waiting_for_first_token"] = False
                    state["first_token_time"] = time.time_ns()
                state["accumulated_response"] += delta.content
            
            # Handle finish_reason
            if hasattr(item.data.choices[0], "finish_reason") and item.data.choices[0].finish_reason:
                state["finish_reason"] = item.data.choices[0].finish_reason
                state["stream_closed_time"] = time.time_ns()
        
        # Handle usage information if available
        if hasattr(item, "data") and hasattr(item.data, "usage") and item.data.usage:
            state["token_usage"] = item.data.usage
            if not state["stream_closed_time"]:
                state["stream_closed_time"] = time.time_ns()

    except Exception as e:
        logger.warning(
            "Warning: Error occurred while processing Mistral stream item: %s",
            str(e),
        )
    finally:
        state["accumulated_temp_list"].append(item)


def _create_span_result(state, stream_start_time):
    """Create the span result object for Mistral streaming."""
    return SimpleNamespace(
        type="stream",
        timestamps={
            "data.input": int(stream_start_time),
            "data.output": int(state["first_token_time"]),
            "metadata": int(state["stream_closed_time"] or time.time_ns()),
        },
        role=state["role"],
        output_text=state["accumulated_response"],
        usage=state["token_usage"],
        finish_reason=state["finish_reason"],
    )


def process_stream(to_wrap, response, span_processor):
    """Process Mistral streaming response by patching the response object's iterator methods."""
    if not to_wrap:
        return
    
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

    # Handle synchronous iterator
    if hasattr(response, "__iter__") and not hasattr(response, "__aiter__"):
        original_next = response.__next__
        
        def new_next(self):
            try:
                # Get the next item from the original iterator
                item = original_next()
                _process_stream_item(item, state)
                return item
            except StopIteration:
                # Iterator is exhausted, process span result
                if span_processor:
                    ret_val = _create_span_result(state, stream_start_time)
                    span_processor(ret_val)
                raise

        # Patch the __next__ method
        patch_instance_method(response, "__next__", new_next)

    # Handle asynchronous iterator
    if hasattr(response, "__aiter__"):
        original_anext = response.__anext__
        
        async def new_anext(self):
            try:
                # Get the next item from the original async iterator
                item = await original_anext()
                _process_stream_item(item, state)
                return item
            except StopAsyncIteration:
                # Async iterator is exhausted, process span result
                if span_processor:
                    ret_val = _create_span_result(state, stream_start_time)
                    span_processor(ret_val)
                raise

        # Patch the __anext__ method
        patch_instance_method(response, "__anext__", new_anext)


MISTRAL_INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    # "is_auto_close": lambda kwargs: False,
    # "response_processor": process_stream,
    "attributes": [
        [
            {
                "_comment": "provider type ,name , deployment , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: 'inference.mistral'
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: "mistral"
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: "https://api.mistral.ai"
            }
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.llm.' + resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
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
                    "_comment": "this is result from LLM, works for streaming and non-streaming",
                    "attribute": "response",
                    "accessor": lambda arguments: (
                        # Handle streaming: combine chunks if result is iterable and doesn't have 'choices'
                        _helper.extract_assistant_message(
                            {"result": list(arguments["result"])}
                            if hasattr(arguments.get("result"), "__iter__") and not hasattr(arguments.get("result"), "choices")
                            else arguments
                        )
                    )
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM, includes token counts",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(
                        arguments.get("result"),
                        include_token_counts=True  # new flag for streaming handling
                    )
                },
                {
                    "_comment": "finish reason from Anthropic response",
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(arguments)
                },
                {
                    "_comment": "finish type mapped from finish reason",
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(_helper.extract_finish_reason(arguments))
                }
            ]
        }
    ]
}

MISTRAL_INFERENCE_STREAM = {
    "type": SPAN_TYPES.INFERENCE,
    "is_auto_close": lambda kwargs: False,
    "response_processor": process_stream,
    "attributes": [
        [
            {
                "_comment": "provider type ,name , deployment , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: 'inference.mistral'
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: "mistral"
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: "https://api.mistral.ai"
            }
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.llm.' + resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
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
                    "_comment": "this is result from LLM, works for streaming and non-streaming",
                    "attribute": "response",
                    "accessor": lambda arguments: (
                        # Handle streaming: combine chunks if result is iterable and doesn't have 'choices'
                        _helper.extract_assistant_message(
                            arguments
                        )
                    )
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM, includes token counts",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(
                        arguments.get("result"),
                        include_token_counts=True  # new flag for streaming handling
                    )
                },
                {
                    "_comment": "finish reason from Anthropic response",
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(arguments)
                },
                {
                    "_comment": "finish type mapped from finish reason",
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(_helper.extract_finish_reason(arguments))
                },
                {
                    "attribute": "inference_sub_type",
                    "accessor": lambda arguments: _helper.agent_inference_type(arguments)
                }
            ]
        }
    ]
}

MISTRAL_INFERENCE_STREAM = {
    "type": SPAN_TYPES.INFERENCE,
    "is_auto_close": lambda kwargs: False,
    "response_processor": process_stream,
    "attributes": [
        [
            {
                "_comment": "provider type ,name , deployment , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: 'inference.mistral'
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: "mistral"
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: "https://api.mistral.ai"
            }
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.llm.' + resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
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
                    "_comment": "this is result from LLM, works for streaming and non-streaming",
                    "attribute": "response",
                    "accessor": lambda arguments: (
                        # Handle streaming: combine chunks if result is iterable and doesn't have 'choices'
                        _helper.extract_assistant_message(
                            arguments
                        )
                    )
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM, includes token counts",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(
                        arguments.get("result"),
                        include_token_counts=True  # new flag for streaming handling
                    )
                },
                {
                    "_comment": "finish reason from Anthropic response",
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(arguments)
                },
                {
                    "_comment": "finish type mapped from finish reason",
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(_helper.extract_finish_reason(arguments))
                },
                {
                    "attribute": "inference_sub_type",
                    "accessor": lambda arguments: _helper.agent_inference_type(arguments)
                }
            ]
        }
    ]
}