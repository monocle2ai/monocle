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


def extract_messages(kwargs):
    """Extract system and user messages"""
    try:
        messages = []
        if len(kwargs['messages']) >0:
            messages = next((msg['content'] for msg in kwargs['messages'] if msg['role'] == 'user'), None)

        return messages
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(response):
    try:
        if hasattr(response,"choices") and len(response.choices) >0:
            if hasattr(response.choices[0],"message"):
                return response.choices[0].message.content
    except (IndexError, AttributeError) as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return None

def extract_provider_name(instance):
    provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)


def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    if inference_endpoint.is_none() and "meta" in instance.client.__dict__:
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    return inference_endpoint.unwrap_or(extract_provider_name(instance))

def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias:
        if i in my_map.keys():
            return my_map[i]
    return None


def update_input_span_events(args):
    return args[0] if len(args) > 0 else ""


def update_output_span_events(results):
    output_arg_text = " ".join([doc.page_content for doc in results if hasattr(doc, 'page_content')])
    if len(output_arg_text) > 100:
        output_arg_text = output_arg_text[:100] + "..."
    return output_arg_text


def update_span_from_llm_response(response):
    meta_dict = {}
    if response is not None and hasattr(response, "usage"):
        if hasattr(response, "usage") and response.usage is not None:
            token_usage = response.usage
        else:
            response_metadata = response.response_metadata
            token_usage = response_metadata.get("token_usage")
        if token_usage is not None:
            meta_dict.update(
                {"completion_tokens": getattr(response.usage, "completion_tokens", None)})
            meta_dict.update({"prompt_tokens": getattr(response.usage, "prompt_tokens", None)})
            meta_dict.update({"total_tokens": getattr(response.usage, "total_tokens", None)})
    return meta_dict

