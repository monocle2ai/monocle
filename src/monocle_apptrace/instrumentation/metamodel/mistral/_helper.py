"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import json
import logging
from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    get_keys_as_tuple,
    get_nested_value,
    get_status_code,
    try_option,
    get_exception_message,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import map_anthropic_finish_reason_to_finish_type
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY, INFERENCE_AGENT_DELEGATION, INFERENCE_TURN_END, INFERENCE_TOOL_CALL


logger = logging.getLogger(__name__)

def extract_provider_name(instance):
    provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)

def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    if inference_endpoint.is_none() and "meta" in instance.client.__dict__:
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    return inference_endpoint.unwrap_or(extract_provider_name(instance))

def dummy_method(arguents):
    pass

def extract_messages(kwargs):
    """Extract system and user messages"""
    try:
        messages = []
        if "system" in kwargs and isinstance(kwargs["system"], str):
            messages.append({"system": kwargs["system"]})
        if 'messages' in kwargs and len(kwargs['messages']) >0:
            for msg in kwargs['messages']:
                if msg.get('content') and msg.get('role'):
                    messages.append({msg['role']: msg['content']})
        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []

def get_exception_status_code(arguments):
    exc = arguments.get("exception")
    if exc is not None and hasattr(exc, "status_code"):
        if exc.status_code == 401:
            return "unauthorized"
        elif exc.status_code == 403:
            return "forbidden"
        elif exc.status_code == 404:
            return "not_found"
        else:
            return str(exc.status_code)
    elif exc is not None:
        return "error"
    else:
        return "success"


def extract_assistant_message(arguments):
    """
    Extract the assistant message from a Mistral response or stream chunks.
    Returns a JSON string like {"assistant": "<text>"}.
    """
    try:
        result = arguments.get("result")
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
                # Each chunk may have delta attribute
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "content"):
                    content.append(chunk.delta.content or "")
            return get_json_dumps({"assistant": "".join(content)})

        return ""

    except Exception as e:
        logger.warning("Warning in extract_assistant_message: %s", str(e))
        return ""



def update_span_from_llm_response(response):
    meta_dict = {}
    if response is not None and hasattr(response, "usage"):
        if hasattr(response, "usage") and response.usage is not None:
            token_usage = response.usage
        else:
            response_metadata = response.response_metadata
            token_usage = response_metadata.get("token_usage")
        if token_usage is not None:
            meta_dict.update({"completion_tokens": getattr(response.usage, "output_tokens", 0)})
            meta_dict.update({"prompt_tokens": getattr(response.usage, "input_tokens", 0)})
            meta_dict.update({"total_tokens": getattr(response.usage, "input_tokens", 0)+getattr(response.usage, "output_tokens", 0)})
    return meta_dict

def extract_finish_reason(arguments):
    """Extract stop_reason from Anthropic response (Claude)."""
    try:
        #streaming
        if hasattr(response, "stop_reason") and response.stop_reason:
            return response.stop_reason

        # Arguments may be a dict with 'result' or just the response object
        response = arguments.get("result") if isinstance(arguments, dict) else arguments
        if response is not None and hasattr(response, "stop_reason"):
            return response.stop_reason
        
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None
    return None

def map_finish_reason_to_finish_type(finish_reason):
    """Map Anthropic stop_reason to finish_type, similar to OpenAI mapping."""
    return map_anthropic_finish_reason_to_finish_type(finish_reason)

def agent_inference_type(arguments):
    """Extract agent inference type from Anthropic response"""
    try:
        status = get_status_code(arguments)
        if status == 'success' or status == 'completed':
            response = arguments["result"]
            
            # Check if stop_reason indicates tool use
            if hasattr(response, "stop_reason") and response.stop_reason == "tool_use":
                # Check if this is agent delegation by looking at tool names
                if hasattr(response, "content") and response.content:
                    agent_prefix = get_value(AGENT_PREFIX_KEY)
                    for content_block in response.content:
                        if (hasattr(content_block, "type") and 
                            content_block.type == "tool_use" and
                            hasattr(content_block, "name")):
                            tool_name = content_block.name
                            if agent_prefix and tool_name.startswith(agent_prefix):
                                return INFERENCE_AGENT_DELEGATION
                    # If we found tool use but no agent delegation, it's a regular tool call
                    return INFERENCE_TOOL_CALL
            
            # Fallback: check the extracted message for tool content
            assistant_message = extract_assistant_message(arguments)
            if assistant_message:
                try:
                    message = json.loads(assistant_message)
                    if message and isinstance(message, dict):
                        assistant_content = message.get("assistant", "")
                        if assistant_content:
                            agent_prefix = get_value(AGENT_PREFIX_KEY)
                            if agent_prefix and agent_prefix in assistant_content:
                                return INFERENCE_AGENT_DELEGATION
                except (json.JSONDecodeError, TypeError):
                    # If JSON parsing fails, fall back to string analysis
                    agent_prefix = get_value(AGENT_PREFIX_KEY)
                    if agent_prefix and agent_prefix in assistant_message:
                        return INFERENCE_AGENT_DELEGATION
        
        return INFERENCE_TURN_END
    except Exception as e:
        logger.warning("Warning: Error occurred in agent_inference_type: %s", str(e))
        return INFERENCE_TURN_END