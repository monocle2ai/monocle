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
    get_exception_message,
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

def get_exception_status_code(arguments):
    if arguments['exception'] is not None and hasattr(arguments['exception'], 'status_code') and arguments['exception'].status_code is not None:
        return arguments['exception'].status_code
    elif arguments['exception'] is not None:
        return 'error'
    else:
        return 'success'

def get_status_code(arguments):
    if arguments["exception"] is not None:
        return get_exception_status_code(arguments)
    elif hasattr(arguments["result"], "status"):
        return arguments["result"].status
    else:
        return 'success'

def extract_assistant_message(arguments):
    try:
        status = get_status_code(arguments)
        response: str = ""
        if status == 'success':
            if arguments['result'] is not None and hasattr(arguments['result'],"content") and len(arguments['result'].content) >0:
                if hasattr(arguments['result'].content[0],"text"):
                    response = arguments['result'].content[0].text
        else:
            if arguments["exception"] is not None:
                response = get_exception_message(arguments)
            elif hasattr(arguments["result"], "error"):
                response = arguments["result"].error
        return response
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