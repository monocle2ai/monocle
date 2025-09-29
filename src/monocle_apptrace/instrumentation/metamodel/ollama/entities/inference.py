import logging
import random
import time
from types import SimpleNamespace
from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.ollama import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import (
    get_error_message,
    patch_instance_method,
    resolve_from_alias,
)

logger = logging.getLogger(__name__)


def _process_stream_item(item, state):
    """Process a single stream item and update state for Ollama."""
    try:
        # Handle Ollama streaming response structure
        if hasattr(item, "message") and hasattr(item.message, "content"):
            if state["waiting_for_first_token"]:
                state["waiting_for_first_token"] = False
                state["first_token_time"] = time.time_ns()
            
            if item.message.content:
                state["accumulated_response"] += item.message.content
                if hasattr(item.message, "role"):
                    state["role"] = item.message.role
                    
        elif hasattr(item, "response"):
            # Handle GenerateResponse streaming
            if state["waiting_for_first_token"]:
                state["waiting_for_first_token"] = False
                state["first_token_time"] = time.time_ns()
            
            if item.response:
                state["accumulated_response"] += item.response
                
        # Check for completion
        if hasattr(item, "done") and item.done:
            state["stream_closed_time"] = time.time_ns()
            if hasattr(item, "done_reason"):
                state["finish_reason"] = item.done_reason
            # Extract usage information if available
            if hasattr(item, "usage"):
                state["token_usage"] = item.usage
                
        # Handle dict-based responses
        elif isinstance(item, dict):
            if "message" in item and "content" in item["message"]:
                if state["waiting_for_first_token"]:
                    state["waiting_for_first_token"] = False
                    state["first_token_time"] = time.time_ns()
                
                if item["message"]["content"]:
                    state["accumulated_response"] += item["message"]["content"]
                    state["role"] = item["message"].get("role", "assistant")
                    
            elif "response" in item:
                if state["waiting_for_first_token"]:
                    state["waiting_for_first_token"] = False
                    state["first_token_time"] = time.time_ns()
                
                if item["response"]:
                    state["accumulated_response"] += item["response"]
                    
            if item.get("done"):
                state["stream_closed_time"] = time.time_ns()
                if "done_reason" in item:
                    state["finish_reason"] = item["done_reason"]

        state["accumulated_temp_list"].append(item)
        
    except Exception as e:
        logger.warning("Warning: Error occurred while processing stream item: %s", str(e))


def _create_span_result(state, stream_start_time):
    """Create the span result object for Ollama streaming."""
    return SimpleNamespace(
        type="stream",
        timestamps={
            "role": state["role"],
            "data.input": int(stream_start_time),
            "data.output": int(state["first_token_time"]),
            "metadata": int(state["stream_closed_time"] or time.time_ns()),
        },
        output_text=state["accumulated_response"],
        tools=state.get("tools"),  # Ollama may support tools in future
        usage=state["token_usage"],
        finish_reason=state["finish_reason"],
    )


def process_stream(to_wrap, response, span_processor):
    """Process Ollama streaming responses using OpenAI-style approach."""
    print(f"DEBUG: process_stream called with response type: {type(response)}")
    print(f"DEBUG: to_wrap={to_wrap}, span_processor={span_processor}")
    
    # For raw generators (Ollama), we need to consume the entire stream here
    # since we can't patch the generator methods like OpenAI does
    if to_wrap and hasattr(response, 'gi_frame'):  # Generator object
        print("DEBUG: Processing raw generator stream")
        
        stream_start_time = time.time_ns()
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
        
        # Consume the entire generator and collect items
        collected_items = []
        try:
            for item in response:
                print(f"DEBUG: Processing stream item: {item}")
                _process_stream_item(item, state)
                collected_items.append(item)
            
            print("DEBUG: Stream completed, calling span_processor")
            state["stream_closed_time"] = time.time_ns()
            if span_processor:
                ret_val = _create_span_result(state, stream_start_time)
                span_processor(ret_val)
                
        except Exception as e:
            print(f"DEBUG: Error processing stream: {e}")
            raise
        
        print(f"DEBUG: Collected {len(collected_items)} items")
        # Return a new generator that yields the collected items
        return (item for item in collected_items)
    
    # Handle class-based streaming (like OpenAI)
    elif to_wrap and hasattr(response, "__iter__"):
        stream_start_time = time.time_ns()
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
        
        original_iter = response.__iter__

        def new_iter(self):
            for item in original_iter():
                _process_stream_item(item, state)
                yield item

            if span_processor:
                ret_val = _create_span_result(state, stream_start_time)
                span_processor(ret_val)

        patch_instance_method(response, "__iter__", new_iter)

    elif to_wrap and hasattr(response, "__aiter__"):
        stream_start_time = time.time_ns()
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
    "is_auto_close": lambda kwargs: kwargs.get("stream", False) is False,  # Use OpenAI-style streaming detection
    "response_processor": process_stream,
    "attributes": [
        [
            {
                "_comment": "provider type, name, deployment, inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: "inference."
                + (_helper.get_inference_type(arguments["instance"]))
                or "ollama",
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
                + resolve_from_alias(
                    arguments["kwargs"], ["model"]
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
                },
                {
                    "attribute": "parameters",
                    "accessor": lambda arguments: resolve_from_alias(
                        arguments["kwargs"],
                        [
                            "temperature",
                            "top_p",
                            "top_k", 
                            "num_predict",
                            "repeat_penalty",
                            "seed",
                            "stop",
                            "format",
                            "options",
                            "keep_alive"
                        ],
                    ),
                },
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
                    "_comment": "finish reason from Ollama response",
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
