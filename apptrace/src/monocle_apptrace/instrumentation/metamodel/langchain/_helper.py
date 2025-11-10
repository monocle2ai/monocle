"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY, INFERENCE_AGENT_DELEGATION, INFERENCE_TURN_END, INFERENCE_TOOL_CALL
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
    get_exception_message,
    get_status_code,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import map_langchain_finish_reason_to_finish_type
from contextlib import suppress

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
            else:
                for msg in args[0]:
                    if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.content:
                        messages.append({msg.type: msg.content})
                    elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                        messages.append({msg.type: get_json_dumps(msg.tool_calls)})
        return [get_json_dumps(d) for d in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []
def agent_inference_type(arguments):
    """Extract agent inference type from arguments."""
    try:
        if get_value(AGENT_PREFIX_KEY):
            agent_prefix = get_value(AGENT_PREFIX_KEY)
            if hasattr(arguments['result'], "tool_calls") and arguments['result'].tool_calls:
                tool_call = arguments['result'].tool_calls[0] if arguments['result'].tool_calls else None
                if tool_call and 'name' in tool_call and tool_call["name"].startswith(agent_prefix):
                    return INFERENCE_AGENT_DELEGATION
                else:
                    return INFERENCE_TOOL_CALL
        return INFERENCE_TURN_END
            
    except Exception as e:
        logger.warning("Warning: Error occurred in agent_inference_type: %s", str(e))
        return None

def extract_assistant_message(arguments):
    status = get_status_code(arguments)
    messages = []
    role = "assistant"
    if status == 'success':
        if isinstance(arguments['result'], str):
            messages.append({role: arguments['result']})
        elif hasattr(arguments['result'], "content") and arguments['result'].content != "":
            role = arguments['result'].type if hasattr(arguments['result'], 'type') else role
            messages.append({role: arguments['result'].content})
        elif hasattr(arguments['result'], "message") and hasattr(arguments['result'].message, "content") and arguments['result'].message.content != "":
            role = arguments['result'].type if hasattr(arguments['result'], 'type') else role
            messages.append({role: arguments['result'].message.content})
        elif hasattr(arguments['result'], "tool_calls"):
            role = arguments['result'].type if hasattr(arguments['result'], 'type') else role
            messages.append({role: arguments['result'].tool_calls[0]})
    else:
        if arguments["exception"] is not None:
            messages.append({role: get_exception_message(arguments)})
        elif hasattr(arguments["result"], "error"):
            return arguments["result"].error
    return get_json_dumps(messages[0]) if messages else ""

def extract_provider_name(instance):
    provider_url: Option[str] = Option(None)
    if hasattr(instance, 'client'):
        provider_url: Option[str] = try_option(getattr, instance.client, 'universe_domain')
    if hasattr(instance,'client') and hasattr(instance.client, '_client') and hasattr(instance.client._client, 'base_url'):
        # If the client has a base_url, extract the host from it
        provider_url: Option[str] = try_option(getattr, instance.client._client.base_url, 'host')
    if hasattr(instance, '_client') and hasattr(instance._client, 'base_url'):
        provider_url = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)


def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = None
    # instance.client.meta.endpoint_url
    if hasattr(instance, 'client') and hasattr(instance.client, 'transport'):
        inference_endpoint: Option[str] = try_option(getattr, instance.client.transport, 'host')

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

def extract_finish_reason(arguments):
    """Extract finish_reason from LangChain response."""
    try:
        # Handle exception cases first
        if arguments.get("exception") is not None:
            # If there's an exception, it's typically an error finish type
            return "error"
        
        response = arguments.get("result")
        if response is None:
            return None
            
        # Check various possible locations for finish_reason in LangChain responses
        
        # Direct finish_reason attribute
        if hasattr(response, "finish_reason") and response.finish_reason:
            return response.finish_reason
            
        # Response metadata (common in LangChain)
        if hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata
            if isinstance(metadata, dict):
                # Check for finish_reason in metadata
                if "finish_reason" in metadata:
                    return metadata["finish_reason"]
                # Check for stop_reason (Anthropic style through LangChain)
                if "stop_reason" in metadata:
                    return metadata["stop_reason"]
                # Check for other common finish reason keys
                for key in ["completion_reason", "end_reason", "status"]:
                    if key in metadata:
                        return metadata[key]
        
        # Check if response has generation_info (some LangChain models)
        if hasattr(response, "generation_info") and response.generation_info:
            gen_info = response.generation_info
            if isinstance(gen_info, dict):
                for key in ["finish_reason", "stop_reason", "completion_reason"]:
                    if key in gen_info:
                        return gen_info[key]
        
        # Check if response has llm_output (batch responses)
        if hasattr(response, "llm_output") and response.llm_output:
            llm_output = response.llm_output
            if isinstance(llm_output, dict):
                for key in ["finish_reason", "stop_reason"]:
                    if key in llm_output:
                        return llm_output[key]
        
        # For AIMessage responses, check additional_kwargs
        if hasattr(response, "additional_kwargs") and response.additional_kwargs:
            kwargs = response.additional_kwargs
            if isinstance(kwargs, dict):
                for key in ["finish_reason", "stop_reason"]:
                    if key in kwargs:
                        return kwargs[key]
        
        # For generation responses with choices (similar to OpenAI structure)
        if hasattr(response, "generations") and response.generations:
            generations = response.generations
            if isinstance(generations, list) and len(generations) > 0:
                for generation in generations:
                    if hasattr(generation, "generation_info") and generation.generation_info:
                        gen_info = generation.generation_info
                        if isinstance(gen_info, dict):
                            for key in ["finish_reason", "stop_reason"]:
                                if key in gen_info:
                                    return gen_info[key]
        
        # If no specific finish reason found, infer from status
        status_code = get_status_code(arguments)
        if status_code == 'success':
            return "stop"  # Default success finish reason
        elif status_code == 'error':
            return "error"
            
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None
    
    return None


def map_finish_reason_to_finish_type(finish_reason):
    """Map LangChain finish_reason to finish_type."""
    return map_langchain_finish_reason_to_finish_type(finish_reason)


def _get_first_tool_call(response):
    """Helper function to extract the first tool call from various LangChain response formats"""

    with suppress(AttributeError, IndexError, TypeError):
        if response.tool_calls:
            return response.tool_calls[0]

    return None

def extract_tool_name(arguments):
    """Extract tool name from LangChain response when finish_type is tool_call"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_call = _get_first_tool_call(arguments["result"])
        if not tool_call:
            return None

        # Try different name extraction approaches
        for getter in [
            lambda tc: tc['name'],  # dict with name key
        ]:
            try:
                return getter(tool_call)
            except (KeyError, AttributeError, TypeError):
                continue
                            
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_name: %s", str(e))
    
    return None

def extract_tool_type(arguments):
    """Extract tool type from LangChain response when finish_type is tool_call"""
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