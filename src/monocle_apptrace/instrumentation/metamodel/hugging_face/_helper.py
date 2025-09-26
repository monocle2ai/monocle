'''"""
This module provides utility functions for extracting system, user,
and assistant messages from Hugging Face inference API responses.


import json
import logging
from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    get_status_code,
    try_option,
)

logger = logging.getLogger(__name__)


def extract_provider_name(instance):
    """Return the Hugging Face API host"""
    provider_url: Option[str] = try_option(getattr, instance._client, 'base_url')
    return provider_url.unwrap_or(None)


def update_input_span_events(kwargs):
    """Extract input text for spans"""
    if "inputs" in kwargs and isinstance(kwargs["inputs"], list):
        return " | ".join(kwargs["inputs"])
    elif "inputs" in kwargs and isinstance(kwargs["inputs"], str):
        return kwargs["inputs"]
    return ""


def update_output_span_events(results):
    """Extract output text for spans"""
    try:
        if hasattr(results, "generated_text"):
            output = results.generated_text
            if len(output) > 200:
                output = output[:200] + "..."
            return output
        elif isinstance(results, list):
            # Hugging Face streaming / batch outputs
            output_texts = [getattr(r, "generated_text", "") for r in results]
            output = " | ".join(output_texts)
            if len(output) > 200:
                output = output[:200] + "..."
            return output
    except Exception as e:
        logger.warning("Error in update_output_span_events: %s", str(e))
    return ""


def extract_inference_endpoint(instance):
    """Return the API endpoint used"""
    endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    return endpoint.unwrap_or(extract_provider_name(instance))


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
        logger.warning("Warning: Error in extract_messages: %s", str(e))
        return []


def get_exception_status_code(arguments):
    """Map exceptions to status codes for spans"""
    exc = arguments.get("exception")
    if exc is not None:
        # Hugging Face returns HTTP errors as exceptions with 'status_code'
        status_code = getattr(exc, "status_code", None)
        if status_code == 401:
            return "unauthorized"
        elif status_code == 403:
            return "forbidden"
        elif status_code == 404:
            return "not_found"
        else:
            return "error"
    return "success"


def extract_assistant_message(arguments):
    """
    Extract the assistant message from a Hugging Face response.
    Returns a JSON string like {"assistant": "<text>"}.
    """
    try:
        result = arguments.get("result") if isinstance(arguments, dict) else arguments
        if result is None:
            return ""

        # Full response object
        if hasattr(result, "generated_text"):
            return get_json_dumps({"assistant": result.generated_text})

        # List of responses (batch or streaming)
        if isinstance(result, list):
            content = []
            for r in result:
                text = getattr(r, "generated_text", "")
                if text:
                    content.append(text)
            return get_json_dumps({"assistant": " ".join(content)})

        return ""

    except Exception as e:
        logger.warning("Warning in extract_assistant_message: %s", str(e))
        return ""


def update_span_from_llm_response(result, include_token_counts=False):
    """Return metadata for spans"""
    tokens = {}
    if include_token_counts:
        tokens = {
            "completion_tokens": getattr(result, "completion_tokens", 0),
            "prompt_tokens": getattr(result, "prompt_tokens", 0),
            "total_tokens": getattr(result, "total_tokens", 0),
        }
    return {**tokens, "inference_sub_type": "turn_end"}


def extract_finish_reason(arguments):
    """Hugging Face doesn't have stop_reason, fallback to 'completed'"""
    return "completed"


def agent_inference_type(arguments):
    """
    Extract agent inference type for Hugging Face.
    Here, generally always a simple turn_end unless delegation logic is implemented.
    """
    try:
        status = get_status_code(arguments)
        if status in ('success', 'completed'):
            assistant_message = extract_assistant_message(arguments)
            if assistant_message:
                # Optional: check for agent delegation in message text
                agent_prefix = get_value("agent_prefix")
                try:
                    message = json.loads(assistant_message)
                    content = message.get("assistant", "")
                    if agent_prefix and agent_prefix in content:
                        return "agent_delegation"
                except Exception:
                    if agent_prefix and agent_prefix in assistant_message:
                        return "agent_delegation"
        return "turn_end"

    except Exception as e:
        logger.warning("Warning in agent_inference_type: %s", str(e))
        return "turn_end"
'''

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

from huggingface_hub import InferenceClient

from monocle_apptrace.instrumentation.metamodel.finish_types import map_hf_finish_reason_to_finish_type

logger = logging.getLogger(__name__)


def get_hf_client(api_key=None, provider="huggingface"):
    """Return a Hugging Face InferenceClient instance."""
    key = api_key or os.environ.get("HF_TOKEN")
    if key is None:
        raise ValueError("No HF API key provided")
    return InferenceClient(provider=provider, api_key=key)


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
        result = arguments.get("result") if isinstance(arguments, dict) else arguments
        if result is None:
            return ""

        # Handle full response
        if hasattr(result, "choices") and result.choices:
            msg_obj = result.choices[0].message
            return get_json_dumps({msg_obj.role: msg_obj.content})

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
    tokens = {
        "completion_tokens": getattr(result.usage, "completion_tokens", 0),
        "prompt_tokens": getattr(result.usage, "prompt_tokens", 0),
        "total_tokens": getattr(result.usage, "total_tokens", 0),
    } if include_token_counts else {}
    # Add other metadata fields like finish_reason, etc.
    return {**tokens, "inference_sub_type": "turn_end"}


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
    try:
        return getattr(result, "finish_reason", None)
    except Exception:
        return None


