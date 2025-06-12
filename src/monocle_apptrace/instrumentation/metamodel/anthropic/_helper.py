"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
)


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
        if 'messages' in kwargs and len(kwargs['messages']) >0:
            for msg in kwargs['messages']:
                if msg.get('content') and msg.get('role'):
                    messages.append({msg['role']: msg['content']})

        return [str(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(response):
    try:
        if response is not None and hasattr(response,"content") and len(response.content) >0:
            if hasattr(response.content[0],"text"):
                return response.content[0].text
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
    if not finish_reason:
        return None
    finish_reason_mapping = {
        "end_turn": "success",         # Natural completion
        "max_tokens": "truncated",     # Hit max_tokens limit
        "stop_sequence": "success",    # Hit user stop sequence
        "tool_use": "success",         # Tool use triggered
        "pause_turn": "success",       # Paused for tool or server action
        "refusal": "refusal",          # Refused for safety/ethics
    }
    return finish_reason_mapping.get(finish_reason, None)