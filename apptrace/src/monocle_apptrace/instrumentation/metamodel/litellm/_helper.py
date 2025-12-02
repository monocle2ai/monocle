"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""
import json
import logging
from monocle_apptrace.instrumentation.common.constants import INFERENCE_TOOL_CALL, INFERENCE_TURN_END, TOOL_TYPE
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    try_option,
    get_exception_message,
    get_status_code,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import map_litellm_finish_reason_to_finish_type
from contextlib import suppress

logger = logging.getLogger(__name__)


def extract_messages(kwargs):
    """Extract system and user messages"""
    try:
        messages = []
        if 'messages' in kwargs and len(kwargs['messages']) > 0:
            for msg in kwargs['messages']:
                if msg.get('content') and msg.get('role'):
                    messages.append({msg['role']: msg['content']})

        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(arguments):
    try:
        messages = []
        status = get_status_code(arguments)
        if status == 'success' or status == 'completed':
            response = arguments["result"]
            if (response is not None and hasattr(response, "choices") and len(response.choices) > 0):
                if hasattr(response.choices[0], "message"):
                    role = (
                        response.choices[0].message.role
                        if hasattr(response.choices[0].message, "role")
                        else "assistant"
                    )
                    messages.append({role: response.choices[0].message.content})
            return get_json_dumps(messages[0]) if messages else ""
        else:
            if arguments["exception"] is not None:
                return get_exception_message(arguments)
            elif hasattr(arguments["result"], "error"):
                return arguments["result"].error

    except (IndexError, AttributeError) as e:
        logger.warning(
            "Warning: Error occurred in extract_assistant_message: %s", str(e)
        )
        return None

def extract_provider_name(url):
    """Extract host from a URL string (e.g., https://api.openai.com/v1/ -> api.openai.com)"""
    if not url:
        return None
    return url.split("//")[-1].split("/")[0]

def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias:
        if i in my_map.keys():
            return my_map[i]
    return None


def update_span_from_llm_response(response):
    meta_dict = {}
    token_usage = None
    if response is not None:
        if token_usage is None and hasattr(response, "usage") and response.usage is not None:
            token_usage = response.usage
        elif token_usage is None and hasattr(response, "response_metadata"):
            token_usage = getattr(response.response_metadata, "token_usage", None) \
                if hasattr(response.response_metadata, "token_usage") \
                else response.response_metadata.get("token_usage", None)
        if token_usage is not None:
            meta_dict.update({"completion_tokens": getattr(token_usage, "completion_tokens", None) or getattr(token_usage, "output_tokens", None)})
            meta_dict.update({"prompt_tokens": getattr(token_usage, "prompt_tokens", None) or getattr(token_usage, "input_tokens", None)})
            meta_dict.update({"total_tokens": getattr(token_usage, "total_tokens")})
    return meta_dict

def agent_inference_type(arguments):
    """Extract agent inference type from LiteLLM response, following OpenAI pattern"""

    finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
    if finish_type == "tool_call":
        return INFERENCE_TOOL_CALL

    return INFERENCE_TURN_END

def extract_finish_reason(arguments):
    """Extract finish_reason from LiteLLM response"""
    try:
        if arguments.get("exception") is not None:
            return "error"
            
        response = arguments.get("result")
        
        # Handle LiteLLM response structure (similar to OpenAI)
        if response is not None and hasattr(response, "choices") and len(response.choices) > 0:
            if hasattr(response.choices[0], "finish_reason"):
                return response.choices[0].finish_reason
                
    except (IndexError, AttributeError) as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None
    return None

def map_finish_reason_to_finish_type(finish_reason):
    """Map LiteLLM finish_reason to finish_type using dedicated LiteLLM mapping"""
    return map_litellm_finish_reason_to_finish_type(finish_reason)

def _get_first_tool_call(response):

    with suppress(AttributeError, IndexError, TypeError):
        if response is not None and hasattr(response, "choices") and len(response.choices) > 0:
            if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "tool_calls"):
                tool_calls = response.choices[0].message.tool_calls
                if tool_calls and len(tool_calls) > 0:
                    return tool_calls[0]

    return None

def extract_tool_name(arguments):
    """Extract tool name from LiteLLM response when finish_type is tool_call"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_call = _get_first_tool_call(arguments["result"])
        if not tool_call:
            return None

        # Try different name extraction approaches
        for getter in [
            lambda tc: tc.function.name,  # dict with name key
        ]:
            try:
                return getter(tool_call)
            except (KeyError, AttributeError, TypeError):
                continue

                    
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_name: %s (type: %s)", str(e), type(e).__name__)
    
    return None

def extract_tool_type(arguments):
    """Extract tool type from LiteLLM response when finish_type is tool_call"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_name = extract_tool_name(arguments)
        if tool_name:
            return TOOL_TYPE

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_type: %s", str(e))
    
    return None