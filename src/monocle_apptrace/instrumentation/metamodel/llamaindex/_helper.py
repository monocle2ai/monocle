"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
from urllib.parse import urlparse
from opentelemetry.sdk.trace import Span
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
)

logger = logging.getLogger(__name__)


def extract_tools(instance):
    tools = []
    if not hasattr(instance, 'state') or not hasattr(instance.state, 'task_dict'):
        return []
    try:
        data = next(iter(instance.state.task_dict.values())).task
    except (AttributeError, StopIteration):
        return []

    if hasattr(data,'extra_state') and 'sources' in data.extra_state:
        for tool_output in data.extra_state['sources']:
            tool_name = tool_output.tool_name
            if tool_name:
                tools.append(tool_name)
    return tools


def extract_messages(args):
    """Extract system and user messages"""
    try:
        messages = []

        def process_message(msg):
            """Processes a single message and extracts relevant information."""
            if hasattr(msg, 'content') and hasattr(msg, 'role'):
                role = getattr(msg.role, 'value', msg.role)
                content = msg.content if role == "system" else extract_query_from_content(msg.content)
                messages.append({role: content})

        if isinstance(args, (list, tuple)) and args:
            for msg in args[0]:
                process_message(msg)
        if isinstance(args, dict):
            for msg in args.get("messages", []):
                process_message(msg)
        if args and isinstance(args, tuple):
            messages.append(args[0])

        return [str(message) for message in messages]

    except Exception as e:
        logger.warning("Error in extract_messages: %s", str(e))
        return []

def extract_assistant_message(response):
    try:
        if isinstance(response, str):
            return [response]
        if hasattr(response, "content"):
            return [response.content]
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return [response.message.content]
        if hasattr(response,"response") and isinstance(response.response, str):
            return [response.response]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return []


def extract_query_from_content(content):
    try:
        query_prefix = "Query:"
        answer_prefix = "Answer:"
        query_start = content.find(query_prefix)
        if query_start == -1:
            return content

        query_start += len(query_prefix)
        answer_start = content.find(answer_prefix, query_start)
        if answer_start == -1:
            query = content[query_start:].strip()
        else:
            query = content[query_start:answer_start].strip()
        return query
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_query_from_content: %s", str(e))
        return ""


def extract_provider_name(instance):
    provider_url = try_option(getattr, instance, 'api_base').and_then(lambda url: urlparse(url).hostname)
    return provider_url


def extract_inference_endpoint(instance):
    inference_endpoint = try_option(getattr, instance._client.sdk_configuration, 'server_url').map(str)
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
    if isinstance(args, tuple):
        return args[0].query_str if len(args) > 0 else ""


def update_output_span_events(results):
    if isinstance(results, list) and len(results) >0:
        output_arg_text = results[0].text
        if len(output_arg_text) > 100:
            output_arg_text = output_arg_text[:100] + "..."
        return output_arg_text


def update_span_from_llm_response(response, instance):
    meta_dict = {}
    if response is not None and hasattr(response, "raw"):
        if response.raw is not None:
            token_usage = response.raw.get("usage") if isinstance(response.raw, dict) else getattr(response.raw, "usage", None)
            if token_usage is not None:
                temperature = instance.__dict__.get("temperature", None)
                meta_dict.update({"temperature": temperature})
                if getattr(token_usage, "completion_tokens", None):
                    meta_dict.update({"completion_tokens": getattr(token_usage, "completion_tokens")})
                if getattr(token_usage, "prompt_tokens", None):
                    meta_dict.update({"prompt_tokens": getattr(token_usage, "prompt_tokens")})
                if getattr(token_usage, "total_tokens", None):
                    meta_dict.update({"total_tokens": getattr(token_usage, "total_tokens")})
    return meta_dict
