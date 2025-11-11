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
from contextlib import suppress


logger = logging.getLogger(__name__)

def extract_provider_name(instance):
    provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)

def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    if inference_endpoint.is_none() and "meta" in instance.client.__dict__:
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    return inference_endpoint.unwrap_or(extract_provider_name(instance))

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
    if arguments['exception'] is not None and hasattr(arguments['exception'], 'status_code') and arguments['exception'].status_code is not None:
        return arguments['exception'].status_code
    elif arguments['exception'] is not None:
        return 'error'
    else:
        return 'success'

def extract_assistant_message(arguments):
    try:
        status = get_status_code(arguments)
        response = arguments["result"]
        if status == 'success':
            messages = []
            role = response.role if hasattr(response, 'role') else "assistant"
            
            # Handle tool use content blocks
            if hasattr(response, "content") and response.content:
                tools = []
                text_content = []
                
                for content_block in response.content:
                    if hasattr(content_block, "type"):
                        if content_block.type == "tool_use":
                            # Extract tool use information
                            tool_info = {
                                "tool_id": getattr(content_block, "id", ""),
                                "tool_name": getattr(content_block, "name", ""),
                                "tool_arguments": getattr(content_block, "input", "")
                            }
                            tools.append(tool_info)
                        elif content_block.type == "text":
                            # Extract text content
                            if hasattr(content_block, "text"):
                                text_content.append(content_block.text)
                
                # If we have tools, add them to the message
                if tools:
                    messages.append({"tools": tools})
                
                # If we have text content, add it to the message
                if text_content:
                    messages.append({role: " ".join(text_content)})
                
                # Fallback to original logic if no content blocks were processed
                if not messages and len(response.content) > 0:
                    if hasattr(response.content[0], "text"):
                        messages.append({role: response.content[0].text})
            
            # Return first message if list is not empty
            return get_json_dumps(messages[0]) if messages else ""
        else:
            if arguments["exception"] is not None:
                return get_exception_message(arguments)
            elif hasattr(arguments["result"], "error"):
                return arguments["result"].error

    except (IndexError, AttributeError) as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return None

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

def _get_first_tool_call(response):
    """Helper function to extract the first tool call from various LangChain response formats"""

    with suppress(AttributeError, IndexError, TypeError):
        if hasattr(response, "content") and response.content:
            for content_block in response.content:
                if hasattr(content_block, "type") and content_block.type == "tool_use":
                    return content_block

    return None

def extract_tool_name(arguments):
    """Extract tool name from Anthropic response when finish_type is tool_call"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_call = _get_first_tool_call(arguments["result"])
        if not tool_call:
            return None

        # Try different name extraction approaches
        for getter in [
            lambda tc: tc.name,     # object with name attribute
        ]:
            try:
                return getter(tool_call)
            except (KeyError, AttributeError, TypeError):
                continue

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_name: %s", str(e))
    
    return None

def extract_tool_type(arguments):
    """Extract tool type from Anthropic response when finish_type is tool_call"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_name = extract_tool_name(arguments)
        if tool_name:
            return "tool.function"
            
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_type: %s", str(e))
    
    return None