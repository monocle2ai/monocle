import os
import json
import logging
from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.constants import (
    AGENT_PREFIX_KEY,
    INFERENCE_AGENT_DELEGATION,
    INFERENCE_TURN_END,
    INFERENCE_TOOL_CALL,
)
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    try_option,
)

from monocle_apptrace.instrumentation.metamodel.finish_types import map_hf_finish_reason_to_finish_type

logger = logging.getLogger(__name__)


def _unwrap_result(payload):
    """Return traced result whether accessor received full arguments dict or raw result."""
    if isinstance(payload, dict) and "result" in payload:
        return payload.get("result")
    return payload

def update_input_span_events(kwargs):
    input_text = ""
    print("DEBUG kwargs:", kwargs)
    if "inputs" in kwargs:
        if isinstance(kwargs["inputs"], list):
            input_text = " | ".join(str(i) for i in kwargs["inputs"])
        else:
            input_text = str(kwargs["inputs"])
    elif "messages" in kwargs:
        input_text = json.dumps(kwargs["messages"])
    return {"input": input_text}  # always a dict with 'input'



def update_output_span_events(result):
    try:
        if hasattr(result, "choices") and result.choices:
            output = [c.message for c in result.choices]
            output_str = json.dumps(output)
            return output_str[:200] + "..." if len(output_str) > 200 else output_str
    except Exception as e:
        logger.warning("Error in update_output_span_events: %s", str(e))
    return ""

def extract_messages(kwargs):
    """Extract system and user messages"""
    try:
        messages = []
        if "system" in kwargs and isinstance(kwargs["system"], str):
            messages.append({"system": kwargs["system"]})
        if 'messages' in kwargs and kwargs['messages']:
            for msg in kwargs['messages']:
                if msg.get('content') and msg.get('role'):
                    messages.append({msg['role']: msg['content']})
        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []

def extract_assistant_message(arguments):
    """
    Extract the assistant message from a Mistral response or stream chunks.
    Returns a JSON string like {"assistant": "<text>"}.
    """
    try:
        result = _unwrap_result(arguments)
        if result is None:
            return ""

        # Handle full response
        if hasattr(result, "choices") and result.choices:
            msg_obj = result.choices[0].message
            role = getattr(msg_obj, "role", "assistant")
            content = getattr(msg_obj, "content", "")
            # Some providers return list content parts.
            if isinstance(content, list):
                normalized_parts = []
                for part in content:
                    if isinstance(part, dict):
                        normalized_parts.append(str(part.get("text", "")))
                    else:
                        normalized_parts.append(str(part))
                content = "".join(normalized_parts)
            return get_json_dumps({role: content})

        # Handle streaming: result might be a list of CompletionEvent chunks
        if isinstance(result, list):
            content = []
            for chunk in result:
                if hasattr(chunk, "data") and hasattr(chunk.data, "choices") and chunk.data.choices:
                    choice = chunk.data.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        content.append(choice.delta.content or "")
            return get_json_dumps({"assistant": "".join(content)})

        return ""

    except Exception as e:
        logger.warning("Warning in extract_assistant_message: %s", str(e))
        return ""
    
def update_span_from_llm_response(result, include_token_counts=False):
    result = _unwrap_result(result)
    usage = getattr(result, "usage", None) if result is not None else None

    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
    else:
        completion_tokens = getattr(usage, "completion_tokens", 0)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

    tokens = {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
    } if include_token_counts else {}
    # Add other metadata fields like finish_reason, etc.
    return {**tokens}


def get_exception_status_code(exc):
    if exc is None:
        return "success"
    code = getattr(exc, "status_code", None)
    if code == 401:
        return "unauthorized"
    elif code == 403:
        return "forbidden"
    elif code == 404:
        return "not_found"
    return "error"

def map_finish_reason_to_finish_type(finish_reason):
    """Map Hugging Face finish_reason to finish_type, similar to OpenAI mapping."""
    return map_hf_finish_reason_to_finish_type(finish_reason)


def agent_inference_type(result):
    """
    Simple agent inference type logic: if message contains AGENT_PREFIX_KEY,
    mark as delegation; otherwise it's a normal turn_end.
    """
    try:
        assistant_msg = extract_assistant_message(result)
        if assistant_msg and AGENT_PREFIX_KEY in assistant_msg:
            return INFERENCE_AGENT_DELEGATION
    except Exception as e:
        logger.warning("Error in agent_inference_type: %s", str(e))
    return INFERENCE_TURN_END


def extract_finish_reason(result):
    result = _unwrap_result(result)
    if result is None:
        return None

    try:
        if hasattr(result, "choices") and result.choices:
            first_choice = result.choices[0]
            if isinstance(first_choice, dict):
                return first_choice.get("finish_reason")
            return getattr(first_choice, "finish_reason", None)

        if isinstance(result, dict):
            return result.get("finish_reason")

        return getattr(result, "finish_reason", None)
    except Exception:
        return None


