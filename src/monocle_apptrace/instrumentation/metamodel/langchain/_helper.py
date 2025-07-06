"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
    get_exception_message,
    get_status_code,
)


logger = logging.getLogger(__name__)


def extract_messages(args):
    """Extract system and user messages"""
    try:
        messages = []
        if args and isinstance(args, (list, tuple)) and hasattr(args[0], 'text'):
            return [args[0].text]
        if args and isinstance(args, (list, tuple)) and len(args) > 0:
            if isinstance(args[0], list) and len(args[0]) > 0:
                first_msg = args[0][0]
                if hasattr(first_msg, 'content') and hasattr(first_msg, 'type') and first_msg.type == "human":
                    return args[0][0].content
            if hasattr(args[0], "messages") and isinstance(args[0].messages, list):
                for msg in args[0].messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        messages.append({msg.type: msg.content})
        return [get_json_dumps(d) for d in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []

def extract_assistant_message(arguments):
    status = get_status_code(arguments)
    # response: str = ""
    messages = []
    role = "assistant"
    if status == 'success':
        if isinstance(arguments['result'], str):
            messages.append({role: arguments['result']})
        if hasattr(arguments['result'], "content"):
            role = arguments['result'].type if hasattr(arguments['result'], 'type') else role
            messages.append({role: arguments['result'].content})
        if hasattr(arguments['result'], "message") and hasattr(arguments['result'].message, "content"):
            messages.append({role: arguments['result'].message.content})
    else:
        if arguments["exception"] is not None:
            messages.append({role: get_exception_message(arguments)})
        elif hasattr(arguments["result"], "error"):
            return arguments["result"].error
    return get_json_dumps(messages[0]) if messages else ""

def extract_provider_name(instance):
    provider_url: Option[str] = Option(None)
    if hasattr(instance,'client') and hasattr(instance.client, '_client') and hasattr(instance.client._client, 'base_url'):
        # If the client has a base_url, extract the host from it
        provider_url: Option[str] = try_option(getattr, instance.client._client.base_url, 'host')
    if hasattr(instance, '_client') and hasattr(instance._client, 'base_url'):
        provider_url = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)


def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = None
    # instance.client.meta.endpoint_url
    if hasattr(instance, 'client') and hasattr(instance.client, 'meta') and hasattr(instance.client.meta, 'endpoint_url'):
        inference_endpoint: Option[str] = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    if hasattr(instance,'client') and hasattr(instance.client, '_client'):
        inference_endpoint: Option[str] = try_option(getattr, instance.client._client, 'base_url').map(str)

    if hasattr(instance,'_client'):
        inference_endpoint = try_option(getattr, instance._client, 'base_url').map(str)

    return inference_endpoint.unwrap_or(extract_provider_name(instance))


def extract_vectorstore_deployment(my_map):
    if isinstance(my_map, dict):
        if '_client_settings' in my_map:
            client = my_map['_client_settings'].__dict__
            host, port = get_keys_as_tuple(client, 'host', 'port')
            if host:
                return f"{host}:{port}" if port else host
        keys_to_check = ['client', '_client']
        host = __get_host_from_map(my_map, keys_to_check)
        if host:
            return host
    else:
        if hasattr(my_map, 'client') and '_endpoint' in my_map.client.__dict__:
            return my_map.client.__dict__['_endpoint']
        host, port = get_keys_as_tuple(my_map.__dict__, 'host', 'port')
        if host:
            return f"{host}:{port}" if port else host
    return None

def __get_host_from_map(my_map, keys_to_check):
    for key in keys_to_check:
        seed_connections = get_nested_value(my_map, [key, 'transport', 'seed_connections'])
        if seed_connections and 'host' in seed_connections[0].__dict__:
            return seed_connections[0].__dict__['host']
    return None

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


def update_span_from_llm_response(response, instance):
    meta_dict = {}
    if response is not None and hasattr(response, "response_metadata"):
        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            token_usage = response.usage_metadata
        else:
            response_metadata = response.response_metadata
            token_usage = response_metadata.get("token_usage")
        if token_usage is not None:
            temperature = instance.__dict__.get("temperature", None)
            meta_dict.update({"temperature": temperature})
            meta_dict.update(
                {"completion_tokens": token_usage.get("completion_tokens") or token_usage.get("output_tokens")})
            meta_dict.update({"prompt_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens")})
            meta_dict.update({"total_tokens": token_usage.get("total_tokens")})
    return meta_dict